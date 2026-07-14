"""
DMD Model - Complete implementation with all loss functions and graph distillation.
Fixed for unaligned mode with dynamic per-modality sequence lengths.
Fixed for PyTorch 2.7+ compatibility with cross-modal attention support.

Components:
1. DMD backbone model (DMD class)
2. HingeLoss (Lmar - Margin Loss)
3. DMDLoss (complete loss: cls + rec + cyc + mar + ort)
4. DistillationKernel (Homo GD and Hetero GD)
5. Imports from dmd_utils for all utility functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# Import all utilities from dmd_utils
from .dmd_utils import (
    to_numpy, squeeze, unsqueeze, is_due, softmax,
    min_cosine, distance_metric, get_segments, 
    get_stats, get_stats_detection, info, warn, err
)

# =============================================================================
# Helper Functions
# =============================================================================

def mkConv1d(in_ch, out_ch, ks, bias=False):
    """Create Conv1D layer with zero padding."""
    return nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=0, bias=bias)


# =============================================================================
# Transformer Encoder (standalone, supports cross-modal attention)
# =============================================================================

try:
    from trains.subNets.transformers_encoder.transformer import TransformerEncoder
except ImportError:
    # Custom standalone TransformerEncoder with cross-modal attention support
    class TransformerEncoder(nn.Module):
        """
        Standalone TransformerEncoder supporting both self-attention and cross-modal attention.
        Supports:
        - Self-attention: forward(x) or forward(x, mask)
        - Cross-attention: forward(query, key, value) or forward(query, key, value, mask)
        """
        def __init__(self, embed_dim, num_heads, layers, attn_dropout, 
                     relu_dropout, res_dropout, embed_dropout, attn_mask):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = layers
            
            # Create transformer encoder layers for self-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=relu_dropout,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers
            )
            
            # Cross-attention layer (MultiheadAttention)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
            
            # Layer norm for cross-attention output
            self.norm = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(embed_dropout)
        
        def forward(self, *args, **kwargs):
            """
            Supports both self-attention and cross-modal attention:
            - Self-attention: forward(x) -> (batch, seq, dim)
            - Cross-attention: forward(query, key, value) -> (batch, tgt_seq, dim)
            """
            if len(args) == 1:
                # Self-attention: forward(x)
                x = args[0]
                return self.transformer(x, src_key_padding_mask=None)
            
            elif len(args) == 2:
                # Self-attention with mask: forward(x, mask)
                x, mask = args
                if mask is not None and mask.dim() == 3:
                    mask = None
                return self.transformer(x, src_key_padding_mask=mask)
            
            elif len(args) == 3:
                # Cross-attention: forward(query, key, value)
                query, key, value = args
                out, _ = self.cross_attn(query, key, value, key_padding_mask=None)
                # Add residual connection with layer norm
                out = self.norm(query + self.dropout(out))
                return out
            
            elif len(args) == 4:
                # Cross-attention with mask: forward(query, key, value, mask)
                query, key, value, mask = args
                if mask is not None and mask.dim() == 3:
                    mask = None
                out, _ = self.cross_attn(query, key, value, key_padding_mask=mask)
                # Add residual connection with layer norm
                out = self.norm(query + self.dropout(out))
                return out
            
            else:
                # Fallback: treat as self-attention
                x = args[0] if args else None
                if x is not None:
                    return self.transformer(x, src_key_padding_mask=None)
                return None


# =============================================================================
# BERT Text Encoder (standalone, no trains dependency)
# =============================================================================

try:
    from trains.subNets import BertTextEncoder
except ImportError:
    class BertTextEncoder(nn.Module):
        """Simple fallback text encoder without BERT."""
        def __init__(self, use_finetune=False, transformers=None, pretrained=None):
            super().__init__()
            self.use_finetune = use_finetune
            
        def forward(self, text):
            if text.dim() == 2:
                return text
            return text


# =============================================================================
# Hinge Loss (Lmar - Margin Loss from DMD paper)
# =============================================================================

class HingeLoss(nn.Module):
    """
    Hinge Loss for feature disentanglement (Lmar - Margin Loss).
    Ensures same-class samples have similar features, different-class samples are separated.
    
    From DMD paper Section 3.4 - Modal Disentanglement Loss.
    """
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1) + 1e-8)
        x_norm = torch.max(x_norm, 1e-8 * torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1) + 1e-8)
        y_norm = torch.max(y_norm, 1e-8 * torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape
        s = feats.repeat(1, B).view(-1, F)
        s_ids = ids.view(B, 1).repeat(1, B)
        t = feats.repeat(B, 1)
        t_ids = ids.view(1, B).repeat(B, 1)
        
        cosine = self.compute_cosine(s, t)
        
        equal_mask = torch.eye(B, dtype=torch.bool)
        s_ids = s_ids[~equal_mask].view(B, B - 1)
        t_ids = t_ids[~equal_mask].view(B, B - 1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B - 1)
        
        sim_mask = (s_ids == t_ids)
        margin = 0.15 * abs(s_ids - t_ids)
        
        loss = 0
        loss_num = 0
        for i in range(B):
            sim_num = sum(sim_mask[i])
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1
        
        if loss_num == 0:
            loss_num = 1
        return loss / loss_num


# =============================================================================
# Complete DMD Loss Function
# =============================================================================

class DMDLoss(nn.Module):
    """
    Complete DMD Loss Function.
    
    From DMD paper Section 3.4:
    - Lcls: Classification loss (main task)
    - Lrec: Reconstruction loss (feature reconstruction)
    - Lcyc: Cycle consistency loss (s -> c -> s reconstruction)
    - Lmar: Margin loss (HingeLoss for feature disentanglement)
    - Lort: Orthogonality loss (s and c should be independent)
    
    Total: L = Lcls + λ1*Lrec + λ2*Lcyc + λ3*Lmar + λ4*Lort
    Paper weights: λ1=0.1, λ2=0.05 (using adaptive for others)
    """
    def __init__(self, lambda_rec=0.1, lambda_cyc=0.05, lambda_mar=0.01, lambda_ort=0.01):
        super(DMDLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_cyc = lambda_cyc
        self.lambda_mar = lambda_mar
        self.lambda_ort = lambda_ort
        self.hinge_loss = HingeLoss()
        
    def compute_rec_loss(self, recon_x, orig_x):
        """Lrec: Reconstruction loss (MSE)."""
        return F.mse_loss(recon_x, orig_x, reduction='mean')
    
    def compute_cyc_loss(self, s, s_r):
        """Lcyc: Cycle consistency loss."""
        return F.mse_loss(s, s_r, reduction='mean')
    
    def compute_ort_loss(self, s_l, s_v, s_a):
        """Lort: Orthogonality loss - modality-specific features should be independent."""
        def flatten(x):
            if x.dim() == 3:
                x = x.reshape(x.size(0), -1)
            return x
        
        s_l_flat = flatten(s_l)
        s_v_flat = flatten(s_v)
        s_a_flat = flatten(s_a)
        
        loss = 0
        for f1, f2 in [(s_l_flat, s_v_flat), (s_l_flat, s_a_flat), (s_v_flat, s_a_flat)]:
            cov = torch.mm(f1.t(), f2) / f1.size(0)
            loss += torch.norm(cov, p='fro') / 3.0
        
        return loss
    
    def forward(self, outputs, labels):
        """
        Compute complete DMD loss.
        
        Args:
            outputs: Dict from DMD.forward() containing all intermediate representations
            labels: Ground truth labels (batch,)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components for logging
        """
        # Classification loss (main task)
        cls_loss = F.cross_entropy(outputs['output_logit'], labels)
        
        total_loss = cls_loss
        loss_dict = {'cls': cls_loss.item()}
        
        # Reconstruction loss (Lrec)
        if self.lambda_rec > 0:
            rec_loss = 0
            for mod in ['l', 'v', 'a']:
                recon = outputs.get(f'recon_{mod}', None)
                orig = outputs.get(f'origin_{mod}', None)
                if recon is not None and orig is not None:
                    rec_loss += self.compute_rec_loss(recon, orig)
            rec_loss = rec_loss / 3
            total_loss = total_loss + self.lambda_rec * rec_loss
            loss_dict['rec'] = rec_loss.item()
        
        # Cycle consistency loss (Lcyc)
        if self.lambda_cyc > 0:
            cyc_loss = 0
            for mod in ['l', 'v', 'a']:
                s_orig = outputs.get(f's_{mod}', None)
                s_recon = outputs.get(f's_{mod}_r', None)
                if s_orig is not None and s_recon is not None:
                    s_o = s_orig.reshape(s_orig.size(0), -1)
                    s_r = s_recon.reshape(s_recon.size(0), -1)
                    cyc_loss += F.mse_loss(s_o, s_r)
            cyc_loss = cyc_loss / 3
            total_loss = total_loss + self.lambda_cyc * cyc_loss
            loss_dict['cyc'] = cyc_loss.item()
        
        # Margin loss (Lmar) - using HingeLoss
        if self.lambda_mar > 0:
            mar_loss = 0
            for mod in ['l', 'v', 'a']:
                proj_s = outputs.get(f'proj_s_{mod}', None)
                if proj_s is not None:
                    mar_loss += self.hinge_loss(labels, proj_s)
            mar_loss = mar_loss / 3
            total_loss = total_loss + self.lambda_mar * mar_loss
            loss_dict['mar'] = mar_loss.item()
        
        # Orthogonality loss (Lort)
        if self.lambda_ort > 0:
            ort_loss = self.compute_ort_loss(
                outputs.get('s_l', None),
                outputs.get('s_v', None),
                outputs.get('s_a', None)
            )
            total_loss = total_loss + self.lambda_ort * ort_loss
            loss_dict['ort'] = ort_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# =============================================================================
# Graph Distillation Kernels (from DMD paper)
# =============================================================================

class DistillationKernel(nn.Module):
    """
    Graph Distillation kernel for both Homo GD and Hetero GD.
    
    - Homo GD: Homogeneous distillation, intra-modality information transfer, using distance_metric
    - Hetero GD: Heterogeneous distillation, inter-modality information transfer, using min_cosine
    
    Calculate the edge weights e_{j->k} for each j.
    """
    def __init__(self, n_classes, hidden_size, gd_size, to_idx, from_idx, 
                 gd_prior, gd_reg, w_losses, metric, alpha=1/8, use_min_cosine=False):
        super(DistillationKernel, self).__init__()
        self.W_logit = nn.Linear(n_classes, gd_size)
        self.W_repr = nn.Linear(hidden_size, gd_size)
        self.W_edge = nn.Linear(gd_size * 4, 1)
        self.gd_size = gd_size
        self.to_idx = to_idx
        self.from_idx = from_idx
        self.alpha = alpha
        self.gd_prior = Variable(torch.FloatTensor(gd_prior))
        self.gd_reg = gd_reg
        self.w_losses = w_losses
        self.metric = metric
        self.use_min_cosine = use_min_cosine  # True for Hetero GD, False for Homo GD

    def forward(self, logits, reprs):
        """
        Args:
            logits: (n_modalities, batch_size, n_classes)
            reprs: (n_modalities, batch_size, hidden_size)
        Returns:
            edges: weights e_{j->k} (n_modalities_from, batch_size)
        """
        n_modalities, batch_size = logits.size()[:2]
        
        z_logits = self.W_logit(logits.view(n_modalities * batch_size, -1))
        z_reprs = self.W_repr(reprs.view(n_modalities * batch_size, -1))
        z = torch.cat((z_logits, z_reprs), dim=1).view(n_modalities, batch_size, self.gd_size * 2)
        
        edges = []
        for j in self.to_idx:
            for i in self.from_idx:
                if i == j:
                    continue
                e = self.W_edge(torch.cat((z[j], z[i]), dim=1))
                edges.append(e)
        
        edges = torch.cat(edges, dim=1)
        edges_origin = edges.sum(0).unsqueeze(0).transpose(0, 1)
        edges = F.softmax(edges * self.alpha, dim=1).transpose(0, 1)
        
        return edges, edges_origin

    def distillation_loss(self, logits, reprs, edges):
        """Calculate graph distillation losses."""
        is_cuda = next(self.parameters()).is_cuda if len(list(self.parameters())) > 0 else False
        prior = self.gd_prior.cuda() if is_cuda else self.gd_prior
        
        loss_reg = (edges.mean(1) - prior).pow(2).sum() * self.gd_reg
        loss_logit, loss_repr = 0, 0
        x = 0
        
        for j in self.to_idx:
            for i, idx in enumerate(self.from_idx):
                if i == j:
                    continue
                w_distill = edges[x] + prior[x]
                
                # Logit distillation (always uses distance_metric)
                loss_logit += self.w_losses[0] * distance_metric(
                    logits[j], logits[idx], self.metric, w_distill)
                
                # Representation distillation
                if self.use_min_cosine:
                    # Hetero GD: use min_cosine
                    loss_repr += self.w_losses[1] * min_cosine(
                        reprs[j], reprs[idx], self.metric, w_distill)
                else:
                    # Homo GD: use distance_metric
                    loss_repr += self.w_losses[1] * distance_metric(
                        reprs[j], reprs[idx], self.metric, w_distill)
                
                x = x + 1
        
        return loss_reg, loss_logit, loss_repr


def get_homo_distillation_kernel(n_classes, hidden_size, gd_size, to_idx, from_idx, 
                                  gd_prior, gd_reg, w_losses, metric, alpha=1/8):
    """Homo GD: Homogeneous distillation, using distance_metric to compute representation distillation loss"""
    return DistillationKernel(n_classes, hidden_size, gd_size, to_idx, from_idx,
                               gd_prior, gd_reg, w_losses, metric, alpha, use_min_cosine=False)


def get_hetero_distillation_kernel(n_classes, hidden_size, gd_size, to_idx, from_idx, 
                                    gd_prior, gd_reg, w_losses, metric, alpha=1/8):
    """Hetero GD: Heterogeneous distillation, using min_cosine to compute representation distillation loss"""
    return DistillationKernel(n_classes, hidden_size, gd_size, to_idx, from_idx,
                               gd_prior, gd_reg, w_losses, metric, alpha, use_min_cosine=True)


# =============================================================================
# DMD Model (Main Model Class)
# =============================================================================

class DMD(nn.Module):
    """
    DMD (Dynamic Modal Fusion with Graph Distillation) backbone model.
    Supports both Aligned and Unaligned modes with dynamic per-modality sequence lengths.
    
    Aligned mode (MOSI): text=audio=vision=50 frames
    Unaligned mode (MOSI): text=50, audio=375, vision=500 frames
    """
    def __init__(self, args):
        super(DMD, self).__init__()
        
        # Handle both dict and Namespace args
        if hasattr(args, 'use_bert'):
            self.use_bert = args.use_bert
        else:
            self.use_bert = args.get('use_bert', False) if isinstance(args, dict) else False
        
        if self.use_bert:
            try:
                self.text_model = BertTextEncoder(
                    use_finetune=getattr(args, 'use_finetune', False),
                    transformers=getattr(args, 'transformers', None),
                    pretrained=getattr(args, 'pretrained', 'bert-base-uncased')
                )
            except Exception:
                self.text_model = None
                self.use_bert = False
        
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.dst_dim = dst_feature_dims
        self.num_heads = nheads
        
        # Per-modality sequence lengths
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        elif args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        else:
            if hasattr(args, 'seq_lens'):
                self.len_l, self.len_a, self.len_v = args.seq_lens
            else:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
        
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        
        # Dropout rates
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        
        # Conv1d kernel sizes
        ks_l = args.conv1d_kernel_size_l
        ks_a = args.conv1d_kernel_size_a
        ks_v = args.conv1d_kernel_size_v
        
        # Effective lengths after conv
        self.eff_len_l = self.len_l - ks_l + 1
        self.eff_len_a = self.len_a - ks_a + 1
        self.eff_len_v = self.len_v - ks_v + 1
        
        # Combined dimensions
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim_l = self.d_l * 3
        combined_dim_v = self.d_v * 3
        combined_dim_a = self.d_a * 3
        # Ensemble: concat of [last_h_l(2*d_l), last_h_v(2*d_v), last_h_a(2*d_a), c_fusion_pooled(3*d_l)]
        combined_dim = 2 * self.d_l + 2 * self.d_v + 2 * self.d_a + 3 * self.d_l
        
        # =====================================================================
        # 1. Initial conv projections
        # =====================================================================
        self.proj_l = mkConv1d(self.orig_d_l, self.d_l, ks_l)
        self.proj_a = mkConv1d(self.orig_d_a, self.d_a, ks_a)
        self.proj_v = mkConv1d(self.orig_d_v, self.d_v, ks_v)
        
        # =====================================================================
        # 2. Modality-specific & invariant encoders
        # =====================================================================
        self.encoder_s_l = mkConv1d(self.d_l, self.d_l, 1)
        self.encoder_s_v = mkConv1d(self.d_v, self.d_v, 1)
        self.encoder_s_a = mkConv1d(self.d_a, self.d_a, 1)
        self.encoder_c = mkConv1d(self.d_l, self.d_l, 1)
        
        # =====================================================================
        # 3. Decoders
        # =====================================================================
        self.decoder_l = mkConv1d(self.d_l * 2, self.d_l, 1)
        self.decoder_v = mkConv1d(self.d_v * 2, self.d_v, 1)
        self.decoder_a = mkConv1d(self.d_a * 2, self.d_a, 1)
        
        # =====================================================================
        # 4. Cosine sim projections (per-modality effective lengths)
        # =====================================================================
        self.proj_cosine_l = nn.Linear(combined_dim_low * self.eff_len_l, combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * self.eff_len_v, combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * self.eff_len_a, combined_dim_low)
        
        # =====================================================================
        # 5. Alignment projections
        # =====================================================================
        self.align_c_l = nn.Linear(combined_dim_low * self.eff_len_l, combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * self.eff_len_v, combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * self.eff_len_a, combined_dim_low)
        
        # =====================================================================
        # 6. Self-attention for c vectors
        # =====================================================================
        self.self_attentions_c_l = self._make_attn(self.d_l, self.attn_dropout)
        self.self_attentions_c_v = self._make_attn(self.d_v, self.attn_dropout_v)
        self.self_attentions_c_a = self._make_attn(self.d_a, self.attn_dropout_a)
        
        # =====================================================================
        # 7. Fusion (cross-modal pooling)
        # =====================================================================
        # Project each modality's pooled representation to common dim (d_l)
        self.proj_c_fusion_l = nn.Linear(self.d_l, self.d_l)
        self.proj_c_fusion_v = nn.Linear(self.d_v, self.d_l)
        self.proj_c_fusion_a = nn.Linear(self.d_a, self.d_l)
        # c_fusion = (batch, 3*d_l) after concat
        self.proj1_c = nn.Linear(combined_dim_l, combined_dim_l)
        self.proj2_c = nn.Linear(combined_dim_l, combined_dim_l)
        self.out_layer_c = nn.Linear(combined_dim_l, 1)
        
        # =====================================================================
        # 8. Homogeneous GD (per-modality)
        # =====================================================================
        self.proj1_l_low = nn.Linear(combined_dim_low * self.eff_len_l, combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * self.eff_len_l)
        self.out_layer_l_low = nn.Linear(combined_dim_low * self.eff_len_l, 1)
        
        self.proj1_v_low = nn.Linear(combined_dim_low * self.eff_len_v, combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * self.eff_len_v)
        self.out_layer_v_low = nn.Linear(combined_dim_low * self.eff_len_v, 1)
        
        self.proj1_a_low = nn.Linear(combined_dim_low * self.eff_len_a, combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * self.eff_len_a)
        self.out_layer_a_low = nn.Linear(combined_dim_low * self.eff_len_a, 1)
        
        # =====================================================================
        # 9. Heterogeneous GD
        # =====================================================================
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, 1)
        
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, 1)
        
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, 1)
        
        # =====================================================================
        # 10. Ensemble
        # =====================================================================
        # weight_l/v/a: input = 2*dim (from trans_*_mem output dim, which doubles input)
        # weight_c: input = 3*d_l (from c_fusion pooled dim = d_l*3)
        self.weight_l = nn.Linear(2 * self.d_l, 2 * self.d_l)
        self.weight_v = nn.Linear(2 * self.d_v, 2 * self.d_v)
        self.weight_a = nn.Linear(2 * self.d_a, 2 * self.d_a)
        self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, 1)
        
        # =====================================================================
        # 11. Cross-modal attention
        # =====================================================================
        self.trans_l_with_a = self._make_attn(self.d_l, self.attn_dropout)
        self.trans_l_with_v = self._make_attn(self.d_l, self.attn_dropout)
        self.trans_a_with_l = self._make_attn(self.d_a, self.attn_dropout_a)
        self.trans_a_with_v = self._make_attn(self.d_a, self.attn_dropout_a)
        self.trans_v_with_l = self._make_attn(self.d_v, self.attn_dropout_v)
        self.trans_v_with_a = self._make_attn(self.d_v, self.attn_dropout_v)
        
        self.trans_l_mem = self._make_attn(self.d_l * 2, self.attn_dropout, layers=3)
        self.trans_a_mem = self._make_attn(self.d_a * 2, self.attn_dropout_a, layers=3)
        self.trans_v_mem = self._make_attn(self.d_v * 2, self.attn_dropout_v, layers=3)

    def _make_attn(self, embed_dim, attn_dropout, layers=None):
        """Create transformer encoder."""
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers) if layers else self.layers,
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=None  # Use None for PyTorch 2.7+ compatibility
        )

    def forward(self, text, audio, video, is_distill=False):
        """Forward pass."""
        # Text encoding (BERT if enabled)
        if self.use_bert and self.text_model is not None:
            text = self.text_model(text)
        
        # (batch, seq, dim) -> (batch, dim, seq)
        x_l = F.dropout(text.transpose(1, 2) if text.dim() == 3 else text, 
                        p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        
        # Project to common dimension
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        # Modality-specific and invariant encoding
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)
        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        
        # Per-modality alignment
        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_v.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_a.size(0), -1))
        
        # Reconstruction
        recon_l = self.decoder_l(torch.cat([s_l, c_l], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_v], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_a], dim=1))
        
        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)
        
        # (batch, dim, seq) -> (batch, seq, dim) for transformer (batch_first=True)
        s_l = s_l.permute(0, 2, 1)
        s_v = s_v.permute(0, 2, 1)
        s_a = s_a.permute(0, 2, 1)
        c_l = c_l.permute(0, 2, 1)
        c_v = c_v.permute(0, 2, 1)
        c_a = c_a.permute(0, 2, 1)
        
        def safe_reshape(x, batch_size):
            return x.transpose(0, 1).reshape(batch_size, -1)
        
        bs = x_l.size(0)
        
        # =====================================================================
        # Homogeneous GD (per-modality effective lengths)
        # =====================================================================
        hs_l_low = safe_reshape(c_l, bs)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low = hs_proj_l_low + hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)
        
        hs_v_low = safe_reshape(c_v, bs)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low = hs_proj_v_low + hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)
        
        hs_a_low = safe_reshape(c_a, bs)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low = hs_proj_a_low + hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)
        
        # =====================================================================
        # Cosine sim (per-modality)
        # =====================================================================
        proj_s_l = self.proj_cosine_l(s_l.reshape(bs, -1))
        proj_s_v = self.proj_cosine_v(s_v.reshape(bs, -1))
        proj_s_a = self.proj_cosine_a(s_a.reshape(bs, -1))
        
        # =====================================================================
        # Self-attention on c vectors (returns last layer output)
        # =====================================================================
        # Cross-modal: mean pooling to (batch, dim) regardless of seq length
        # Compatible with both aligned (same seq) and unaligned (different seq) modes
        c_l_pool = c_l.mean(dim=1)   # (batch, d_l)
        c_v_pool = c_v.mean(dim=1)   # (batch, d_v)
        c_a_pool = c_a.mean(dim=1)   # (batch, d_a)

        # Project each to common dimension and concatenate → (batch, 3*d_l)
        c_fusion = torch.cat([
            self.proj_c_fusion_l(c_l_pool),
            self.proj_c_fusion_v(c_v_pool),
            self.proj_c_fusion_a(c_a_pool)
        ], dim=1)  # (batch, 3*d_l)
        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout, training=self.training))
        c_proj = c_proj + c_fusion
        logits_c = self.out_layer_c(c_proj)  # c_proj is (batch, 3*d_l) already
        
        # =====================================================================
        # Cross-modal attention
        # =====================================================================
        # Cross-attention: forward(query, key, value)
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_ls = self.trans_l_mem(torch.cat([h_l_with_as, h_l_with_vs], dim=2))
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        # Mean pooling across seq dimension (compatible with both aligned and unaligned)
        last_h_l = h_ls.mean(dim=1)  # (batch, 2*d_l)

        h_a_with_ls = self.trans_a_with_l(s_a, s_l, s_l)
        h_a_with_vs = self.trans_a_with_v(s_a, s_v, s_v)
        h_as = self.trans_a_mem(torch.cat([h_a_with_ls, h_a_with_vs], dim=2))
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as.mean(dim=1)  # (batch, 2*d_a)

        h_v_with_ls = self.trans_v_with_l(s_v, s_l, s_l)
        h_v_with_as = self.trans_v_with_a(s_v, s_a, s_a)
        h_vs = self.trans_v_mem(torch.cat([h_v_with_ls, h_v_with_as], dim=2))
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs.mean(dim=1)  # (batch, 2*d_v)
        
        # =====================================================================
        # Heterogeneous GD
        # =====================================================================
        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l.unsqueeze(1)), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high = hs_proj_l_high + last_h_l.unsqueeze(1)
        logits_l_high = self.out_layer_l_high(hs_proj_l_high.squeeze(1))
        
        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v.unsqueeze(1)), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high = hs_proj_v_high + last_h_v.unsqueeze(1)
        logits_v_high = self.out_layer_v_high(hs_proj_v_high.squeeze(1))
        
        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a.unsqueeze(1)), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_high = hs_proj_a_high + last_h_a.unsqueeze(1)
        logits_a_high = self.out_layer_a_high(hs_proj_a_high.squeeze(1))
        
        # =====================================================================
        # Ensemble
        # =====================================================================
        last_h_l_w = torch.sigmoid(self.weight_l(last_h_l))
        last_h_v_w = torch.sigmoid(self.weight_v(last_h_v))
        last_h_a_w = torch.sigmoid(self.weight_a(last_h_a))
        c_fusion_w = torch.sigmoid(self.weight_c(c_proj))  # c_proj is (batch, 3*d_l) already
        last_hs = torch.cat([last_h_l_w, last_h_v_w, last_h_a_w, c_fusion_w], dim=1)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj = last_hs_proj + last_hs
        output = self.out_layer(last_hs_proj)
        
        # =====================================================================
        # Return all intermediate representations
        # =====================================================================
        return {
            # Homogeneous GD outputs
            'logits_l_homo': logits_l_low,
            'logits_v_homo': logits_v_low,
            'logits_a_homo': logits_a_low,
            'repr_l_homo': repr_l_low,
            'repr_v_homo': repr_v_low,
            'repr_a_homo': repr_a_low,
            
            # Original features
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            
            # s vectors (modality-specific)
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            
            # Cosine projections
            'proj_s_l': proj_s_l,
            'proj_s_v': proj_s_v,
            'proj_s_a': proj_s_a,
            
            # c vectors (modality-invariant)
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            
            # Reconstruction
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            
            # Similarity features
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            
            # Heterogeneous GD outputs
            'logits_l_hetero': logits_l_high,
            'logits_v_hetero': logits_v_high,
            'logits_a_hetero': logits_a_high,
            'repr_l_hetero': hs_proj_l_high.squeeze(1),
            'repr_v_hetero': hs_proj_v_high.squeeze(1),
            'repr_a_hetero': hs_proj_a_high.squeeze(1),
            
            # Cross-modal attention outputs
            'last_h_l': last_h_l,
            'last_h_v': last_h_v,
            'last_h_a': last_h_a,
            
            # Fusion output
            'logits_c': logits_c,
            
            # Final ensemble output
            'output_logit': output
        }


# Alias for backward compatibility
DMDModel = DMD
