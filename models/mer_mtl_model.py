"""
MER-MTL Model (DMD-aligned Architecture)
============================================================
Based on DMD baseline (CVPR 2023) with MER-MTL auxiliary tasks.
Main task: 7-class classification + Binary classification (dual heads)
Auxiliary tasks: L_rec, L_cyc, L_mar, L_ort (aligned with DMD)

Supports both ALIGNED and UNALIGNED data with proper attention masking.
Supports two text processing modes:
  - TT (Text Transformer): Conv1d -> Transformer (captures temporal dependencies)
  - MP (Mean Pooling): Conv1d -> MeanPool -> MLP (simpler, no temporal modeling)

Architecture alignment with DMD:
  - Conv1d: 768->50, 5->50, 20->50
  - TT mode: Transformer DST=50, nheads=10, nlevels=4
  - MP mode: MeanPool + MLP projection (skip Transformer for text)
  - Ensemble: s_L + s_A + s_V (150) -> 50 -> emotion heads

Unaligned support:
  - Aligned: all modalities share same sequence length, concatenated for joint processing
  - Unaligned: each modality has independent sequence length, processed separately
    then pooled before ensemble (avoids sequence length mismatch)
============================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TRANSFORMER ENCODER (DMD official implementation with mask support)
# =============================================================================
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half = embedding_dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :].fill_(0)
        return emb

    def forward(self, input):
        """
        Args:
            input: (B, T, D) or (B, T) tensor
        Returns:
            positional embedding: (T, embedding_dim) broadcastable to (B, T, D)
        """
        bsz, seq_len = input.size()[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos not in self.weights:
            self.weights[max_pos] = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        return self.weights[max_pos][self.padding_idx + 1:].detach().to(input.device)


class TransformerEncoder(nn.Module):
    """Transformer encoder with key_padding_mask support for unaligned data"""
    def __init__(self, arch):
        super().__init__()
        self.embed_dim = arch['d_hid']
        self.num_heads = arch['n_heads']
        self.layers = arch['n_levels']
        self.attn_dropout = arch.get('attn_dropout', 0.0)
        self.embedding = SinusoidalPositionalEmbedding(self.embed_dim, padding_idx=0, left_pad=0)
        self.head = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.attn_dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(arch.get('relu_dropout', 0.0)),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Dropout(arch.get('relu_dropout', 0.0)),
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.proj_drop = nn.Dropout(arch.get('embed_dropout', 0.0))

    def forward(self, seq, mask=None):
        """
        Args:
            seq: (B, T, D) input sequence
            mask: (B, T) boolean mask where True = padding position to ignore
                  Used as key_padding_mask in MultiheadAttention (correct API).
        """
        pos = self.embedding(seq)
        seq = seq + pos

        # Use key_padding_mask (B, T) - the correct API for padding in MultiheadAttention
        attn_out, _ = self.head(seq, seq, seq, key_padding_mask=mask)
        seq = seq + attn_out
        seq = self.norm1(seq)

        ff = self.fc(seq)
        seq = seq + self.proj_drop(ff)
        seq = self.norm2(seq)

        if mask is not None:
            seq = seq.masked_fill(mask.unsqueeze(-1), 0.0)

        return seq


# =============================================================================
# DMD ALIGNED FEATURE ENCODER (with mask support + TT/MP text modes)
# =============================================================================
class DMDAlignedEncoder(nn.Module):
    """
    DMD-aligned multimodal encoder with attention mask support.
    Handles both aligned and unaligned data.

    Text processing modes:
      - 'tt' (Text Transformer): Conv1d -> Transformer (temporal modeling)
      - 'mp' (Mean Pooling): Conv1d -> MeanPool -> MLP (no temporal modeling)

    Aligned vs Unaligned:
      - Aligned: all modalities concatenated and processed jointly through Transformer
      - Unaligned: each modality processed independently through Transformer
        (avoids sequence length mismatch in concatenation/slicing)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        feat_dims = args.feature_dims  # [768, 5, 20]
        dst = args.dst_feature_dim_nheads[0]  # 50 (DMD standard)
        nheads = args.dst_feature_dim_nheads[1]  # 10 (DMD standard)
        nlevels = args.nlevels  # 4 (DMD standard)

        # Text processing mode: 'tt' or 'mp'
        self.text_mode = getattr(args, 'text_mode', 'tt')

        # DMD-style Conv1d projections to common dimension
        self.proj_l = nn.Conv1d(feat_dims[0], dst, kernel_size=args.conv1d_kernel_size_l, padding=args.conv1d_kernel_size_l // 2)
        self.proj_a = nn.Conv1d(feat_dims[1], dst, kernel_size=args.conv1d_kernel_size_a, padding=args.conv1d_kernel_size_a // 2)
        self.proj_v = nn.Conv1d(feat_dims[2], dst, kernel_size=args.conv1d_kernel_size_v, padding=args.conv1d_kernel_size_v // 2)

        self.proj_drop = nn.Dropout(args.embed_dropout)

        # Transformer encoder config
        arch = {
            'd_hid': dst,
            'n_heads': nheads,
            'n_levels': nlevels,
            'attn_dropout': args.attn_dropout,
            'relu_dropout': args.relu_dropout,
            'embed_dropout': args.embed_dropout,
        }

        # TT mode: shared Transformer for all modalities
        # MP mode: Transformer only for Audio/Vision, text uses MeanPool + MLP
        if self.text_mode == 'tt':
            self.trans = TransformerEncoder(arch)
        else:  # 'mp'
            # Audio/Vision Transformer (shared)
            self.trans_av = TransformerEncoder(arch)
            # Text MeanPool + MLP projection
            self.text_mp_proj = nn.Sequential(
                nn.Linear(dst, dst * 2),
                nn.ReLU(),
                nn.Dropout(args.embed_dropout),
                nn.Linear(dst * 2, dst),
            )

        self.feat_drop = nn.Dropout(args.text_dropout)

        # Flag for aligned/unaligned mode
        self.need_aligned = getattr(args, 'need_data_aligned', True)

    def _masked_mean(self, x, mask, dim=1):
        """Compute masked mean along dimension"""
        if mask is None:
            return x.mean(dim=dim)
        mask_expand = (~mask).unsqueeze(-1).float()
        x_masked = x * mask_expand
        sum_x = x_masked.sum(dim=dim)
        count = mask_expand.sum(dim=dim).clamp(min=1)
        return sum_x / count

    def forward(self, text_x, audio_x, vision_x, text_mask=None, audio_mask=None, vision_mask=None):
        """
        Args:
            text_x: (B, T_text, 768) - BERT embeddings
            audio_x: (B, T_audio, 5) - acoustic features
            vision_x: (B, T_vision, 20) - visual features
            text_mask: (B, T_text) - padding mask for text (True = padding)
            audio_mask: (B, T_audio) - padding mask for audio
            vision_mask: (B, T_vision) - padding mask for vision

        Returns:
            dict with encoded features and masks for each modality
        """
        B = text_x.size(0)
        T_text = text_x.size(1)
        T_audio = audio_x.size(1)
        T_vision = vision_x.size(1)

        # Conv1d: (B, T, D) -> (B, D, T) -> (B, T, 50)
        # Conv1d with padding=kernel_size//2 preserves sequence length
        L = self.proj_l(text_x.transpose(1, 2)).transpose(1, 2)    # (B, T_text, 50)
        A = self.proj_a(audio_x.transpose(1, 2)).transpose(1, 2)   # (B, T_audio, 50)
        V = self.proj_v(vision_x.transpose(1, 2)).transpose(1, 2)  # (B, T_vision, 50)

        L = self.proj_drop(L)
        A = self.proj_drop(A)
        V = self.proj_drop(V)

        if self.text_mode == 'tt':
            # === TT Mode: Transformer for all modalities ===
            if self.need_aligned:
                # Aligned: concatenate all modalities and process jointly
                fused = torch.cat([L, A, V], dim=1)  # (B, T_text+T_audio+T_vision, 50)

                if text_mask is not None or audio_mask is not None or vision_mask is not None:
                    if text_mask is None:
                        text_mask = torch.zeros((B, T_text), dtype=torch.bool, device=text_x.device)
                    if audio_mask is None:
                        audio_mask = torch.zeros((B, T_audio), dtype=torch.bool, device=text_x.device)
                    if vision_mask is None:
                        vision_mask = torch.zeros((B, T_vision), dtype=torch.bool, device=text_x.device)
                    fused_mask = torch.cat([text_mask, audio_mask, vision_mask], dim=-1)
                else:
                    fused_mask = None

                out = self.trans(fused, mask=fused_mask)  # (B, T_text+T_audio+T_vision, 50)

                # Slice using ACTUAL lengths (not assuming all equal)
                L_enc = out[:, :T_text, :]
                A_enc = out[:, T_text:T_text + T_audio, :]
                V_enc = out[:, T_text + T_audio:, :]
            else:
                # Unaligned: process each modality separately through shared Transformer
                L_enc = self.trans(L, mask=text_mask)
                A_enc = self.trans(A, mask=audio_mask)
                V_enc = self.trans(V, mask=vision_mask)
        else:
            # === MP Mode: Text uses MeanPool, Audio/Vision use Transformer ===
            # Text: MeanPool -> MLP -> expand to (B, T_text, 50)
            L_pooled = self._masked_mean(L, text_mask, dim=1)  # (B, 50)
            L_global = self.text_mp_proj(L_pooled)  # (B, 50)
            L_enc = L_global.unsqueeze(1).expand(-1, T_text, -1).contiguous()  # (B, T_text, 50)

            if self.need_aligned:
                # Aligned: concat A, V and process jointly
                fused_av = torch.cat([A, V], dim=1)  # (B, T_audio+T_vision, 50)

                if audio_mask is not None or vision_mask is not None:
                    if audio_mask is None:
                        audio_mask = torch.zeros((B, T_audio), dtype=torch.bool, device=text_x.device)
                    if vision_mask is None:
                        vision_mask = torch.zeros((B, T_vision), dtype=torch.bool, device=text_x.device)
                    fused_av_mask = torch.cat([audio_mask, vision_mask], dim=-1)
                else:
                    fused_av_mask = None

                out_av = self.trans_av(fused_av, mask=fused_av_mask)

                # Slice using ACTUAL lengths
                A_enc = out_av[:, :T_audio, :]
                V_enc = out_av[:, T_audio:, :]
            else:
                # Unaligned: process A, V separately through shared Transformer
                A_enc = self.trans_av(A, mask=audio_mask)
                V_enc = self.trans_av(V, mask=vision_mask)

        L_enc = self.feat_drop(L_enc)
        A_enc = self.feat_drop(A_enc)
        V_enc = self.feat_drop(V_enc)

        # Apply masks to zero out padding
        if text_mask is not None:
            L_enc = L_enc.masked_fill(text_mask.unsqueeze(-1), 0.0)
        if audio_mask is not None:
            A_enc = A_enc.masked_fill(audio_mask.unsqueeze(-1), 0.0)
        if vision_mask is not None:
            V_enc = V_enc.masked_fill(vision_mask.unsqueeze(-1), 0.0)

        return {
            'L': L_enc,
            'A': A_enc,
            'V': V_enc,
            'text_mask': text_mask,
            'audio_mask': audio_mask,
            'vision_mask': vision_mask,
        }


# =============================================================================
# DMD ALIGNED MULTIMODAL FUSION (Crossmodal + Self-attention)
# =============================================================================
class DMDMultimodalFusion(nn.Module):
    """
    DMD-style crossmodal and self-attention module with key_padding_mask support.
    Generates s_L, s_A, s_V (shared representations) and c_L, c_A, c_V (private).

    Handles different sequence lengths for unaligned data:
      - Crossmodal attention: Q from one modality, K/V from another
        (output shape matches Q, so different K/V lengths are fine)
      - Residual connections: only within same modality (same shape guaranteed)
    """
    def __init__(self, dst_dim=50):
        super().__init__()
        self.dst = dst_dim

        # Crossmodal attention
        self.cross_attn_LA = nn.MultiheadAttention(dst_dim, num_heads=10, dropout=0.1, batch_first=True)
        self.cross_attn_LV = nn.MultiheadAttention(dst_dim, num_heads=10, dropout=0.1, batch_first=True)
        self.cross_attn_AL = nn.MultiheadAttention(dst_dim, num_heads=10, dropout=0.1, batch_first=True)
        self.cross_attn_AV = nn.MultiheadAttention(dst_dim, num_heads=10, dropout=0.1, batch_first=True)
        self.cross_attn_VL = nn.MultiheadAttention(dst_dim, num_heads=10, dropout=0.1, batch_first=True)
        self.cross_attn_VA = nn.MultiheadAttention(dst_dim, num_heads=10, dropout=0.1, batch_first=True)

        # Norm layers
        self.norm1 = nn.LayerNorm(dst_dim)
        self.norm2 = nn.LayerNorm(dst_dim)

        # FFN for each modality
        self.ffn_L = nn.Sequential(nn.Linear(dst_dim, dst_dim * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dst_dim * 2, dst_dim))
        self.ffn_A = nn.Sequential(nn.Linear(dst_dim, dst_dim * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dst_dim * 2, dst_dim))
        self.ffn_V = nn.Sequential(nn.Linear(dst_dim, dst_dim * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dst_dim * 2, dst_dim))

        # Shared projection
        self.shared_proj = nn.Linear(dst_dim, dst_dim)

    def forward(self, L, A, V, text_mask=None, audio_mask=None, vision_mask=None):
        """
        Args:
            L: (B, T_L, D), A: (B, T_A, D), V: (B, T_V, D)
            masks: (B, T_modality) boolean, True = padding
        """
        # Crossmodal attention using key_padding_mask (correct API)
        # Q=L, K=V=A: output shape matches Q (T_L), key_padding_mask for K/V (T_A)
        L2A, _ = self.cross_attn_LA(L, A, A, key_padding_mask=audio_mask)
        L2V, _ = self.cross_attn_LV(L, V, V, key_padding_mask=vision_mask)
        L_cross = self.norm1(L + L2A + L2V)

        A2L, _ = self.cross_attn_AL(A, L, L, key_padding_mask=text_mask)
        A2V, _ = self.cross_attn_AV(A, V, V, key_padding_mask=vision_mask)
        A_cross = self.norm1(A + A2L + A2V)

        V2L, _ = self.cross_attn_VL(V, L, L, key_padding_mask=text_mask)
        V2A, _ = self.cross_attn_VA(V, A, A, key_padding_mask=audio_mask)
        V_cross = self.norm1(V + V2L + V2A)

        # FFN + Residual
        L_out = self.norm2(L_cross + self.ffn_L(L_cross))
        A_out = self.norm2(A_cross + self.ffn_A(A_cross))
        V_out = self.norm2(V_cross + self.ffn_V(V_cross))

        # Generate s (shared) and c (private)
        s_L = self.shared_proj(L_out)
        s_A = self.shared_proj(A_out)
        s_V = self.shared_proj(V_out)

        c_L = L_out - s_L
        c_A = A_out - s_A
        c_V = V_out - s_V

        # Apply masks
        if text_mask is not None:
            s_L = s_L.masked_fill(text_mask.unsqueeze(-1), 0.0)
            c_L = c_L.masked_fill(text_mask.unsqueeze(-1), 0.0)
            L_out = L_out.masked_fill(text_mask.unsqueeze(-1), 0.0)
        if audio_mask is not None:
            s_A = s_A.masked_fill(audio_mask.unsqueeze(-1), 0.0)
            c_A = c_A.masked_fill(audio_mask.unsqueeze(-1), 0.0)
            A_out = A_out.masked_fill(audio_mask.unsqueeze(-1), 0.0)
        if vision_mask is not None:
            s_V = s_V.masked_fill(vision_mask.unsqueeze(-1), 0.0)
            c_V = c_V.masked_fill(vision_mask.unsqueeze(-1), 0.0)
            V_out = V_out.masked_fill(vision_mask.unsqueeze(-1), 0.0)

        return {
            's_L': s_L, 's_A': s_A, 's_V': s_V,
            'c_L': c_L, 'c_A': c_A, 'c_V': c_V,
            'L_out': L_out, 'A_out': A_out, 'V_out': V_out,
        }


# =============================================================================
# MER-MTL MODEL (DMD-aligned + 7-class + Binary, TT/MP support)
# =============================================================================
class MERMTLModel(nn.Module):
    """
    MER-MTL Model with DMD-aligned architecture.
    Supports both ALIGNED and UNALIGNED data with attention masking.
    Supports two text processing modes: TT (Text Transformer) and MP (Mean Pooling).

    Main task: 7-class emotion classification + Binary classification (dual heads)
    Auxiliary tasks: L_rec, L_cyc, L_mar, L_ort (aligned with DMD)

    Ensemble strategy:
      - Aligned: concat temporal s_L/s_A/s_V (B,T,150) -> project -> pool -> z
      - Unaligned: pool each modality first (B,50) -> concat (B,150) -> project -> z
        (avoids sequence length mismatch in temporal concatenation)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        dst = args.dst_feature_dim_nheads[0]

        self.need_aligned = getattr(args, 'need_data_aligned', True)
        self.text_mode = getattr(args, 'text_mode', 'tt')

        self.encoder = DMDAlignedEncoder(args)
        self.fusion = DMDMultimodalFusion(dst_dim=dst)

        # Ensemble align projection
        self.align_proj = nn.Sequential(
            nn.Linear(dst * 3, dst),
            nn.ReLU(),
            nn.Dropout(args.output_dropout),
        )

        # Emotion heads
        self.emotion_head_7cls = nn.Linear(dst, 7)
        self.emotion_head_2cls = nn.Linear(dst, 2)

        # Reconstruction heads
        self.recon_l = nn.Linear(dst, args.feature_dims[0])
        self.recon_a = nn.Linear(dst, args.feature_dims[1])
        self.recon_v = nn.Linear(dst, args.feature_dims[2])

        # Private projection for cycle consistency
        self.private_proj = nn.Linear(dst, dst)

        # Similarity heads
        self.sim_head_l = nn.Linear(dst, 1)
        self.sim_head_a = nn.Linear(dst, 1)
        self.sim_head_v = nn.Linear(dst, 1)

    def _masked_mean(self, x, mask, dim=1):
        """Compute masked mean along dimension"""
        if mask is None:
            return x.mean(dim=dim)
        mask_expand = (~mask).unsqueeze(-1).float()
        x_masked = x * mask_expand
        sum_x = x_masked.sum(dim=dim)
        count = mask_expand.sum(dim=dim).clamp(min=1)
        return sum_x / count

    def forward(self, text_x, audio_x, vision_x, text_mask=None, audio_mask=None, vision_mask=None, is_distill=True):
        # Encode features
        enc = self.encoder(text_x, audio_x, vision_x, text_mask, audio_mask, vision_mask)
        L, A, V = enc['L'], enc['A'], enc['V']

        # Crossmodal fusion
        fused = self.fusion(L, A, V, text_mask, audio_mask, vision_mask)
        s_L, s_A, s_V = fused['s_L'], fused['s_A'], fused['s_V']
        c_L, c_A, c_V = fused['c_L'], fused['c_A'], fused['c_V']

        # === Ensemble representation ===
        if self.need_aligned:
            # Aligned: all modalities have same sequence length T
            # Concat temporal sequences -> project -> pool
            ensemble = torch.cat([s_L, s_A, s_V], dim=-1)  # (B, T, 150)
            ensemble = self.align_proj(ensemble)  # (B, T, 50)

            ens_mean = ensemble.mean(dim=1)  # (B, 50)
            ens_max = ensemble.max(dim=1)[0]  # (B, 50)
            z = (ens_mean + ens_max) / 2
        else:
            # Unaligned: modalities have different sequence lengths
            # Pool each modality first -> concat pooled vectors -> project
            s_L_pool = self._masked_mean(s_L, text_mask, dim=1)  # (B, 50)
            s_A_pool = self._masked_mean(s_A, audio_mask, dim=1)  # (B, 50)
            s_V_pool = self._masked_mean(s_V, vision_mask, dim=1)  # (B, 50)
            ensemble = torch.cat([s_L_pool, s_A_pool, s_V_pool], dim=-1)  # (B, 150)
            z = self.align_proj(ensemble)  # (B, 50)

        # Main task outputs
        logits_7cls = self.emotion_head_7cls(z)
        logits_2cls = self.emotion_head_2cls(z)

        if not is_distill:
            return logits_7cls

        # === Auxiliary task outputs ===
        # Reconstruction: pooled encoded -> reconstruct original pooled features
        recon_l = self.recon_l(self._masked_mean(L, text_mask, dim=1))
        recon_a = self.recon_a(self._masked_mean(A, audio_mask, dim=1))
        recon_v = self.recon_v(self._masked_mean(V, vision_mask, dim=1))

        orig_l = self._masked_mean(text_x, text_mask, dim=1)
        orig_a = self._masked_mean(audio_x, audio_mask, dim=1)
        orig_v = self._masked_mean(vision_x, vision_mask, dim=1)

        # Private content features
        c_L_pool = self._masked_mean(c_L, text_mask, dim=1)
        c_A_pool = self._masked_mean(c_A, audio_mask, dim=1)
        c_V_pool = self._masked_mean(c_V, vision_mask, dim=1)

        # Cycle consistency: private -> project -> compare with shared sequence
        s_l_r = self.private_proj(c_L_pool).unsqueeze(-1)
        s_a_r = self.private_proj(c_A_pool).unsqueeze(-1)
        s_v_r = self.private_proj(c_V_pool).unsqueeze(-1)

        # Similarity scores
        c_l_sim = self.sim_head_l(c_L_pool).squeeze(-1)
        c_a_sim = self.sim_head_a(c_A_pool).squeeze(-1)
        c_v_sim = self.sim_head_v(c_V_pool).squeeze(-1)

        return {
            'emotion_logits': logits_7cls,
            'emotion_logits_2cls': logits_2cls,
            'recon_l': recon_l, 'recon_a': recon_a, 'recon_v': recon_v,
            'origin_l': orig_l, 'origin_a': orig_a, 'origin_v': orig_v,
            's_L_raw': self._masked_mean(s_L, text_mask, dim=1),
            's_A_raw': self._masked_mean(s_A, audio_mask, dim=1),
            's_V_raw': self._masked_mean(s_V, vision_mask, dim=1),
            'c_L_raw': c_L_pool, 'c_A_raw': c_A_pool, 'c_V_raw': c_V_pool,
            's_l_r': s_l_r, 's_a_r': s_a_r, 's_v_r': s_v_r,
            's_l_seq': s_L.permute(0, 2, 1), 's_a_seq': s_A.permute(0, 2, 1), 's_v_seq': s_V.permute(0, 2, 1),
            'c_l_sim': c_l_sim, 'c_a_sim': c_a_sim, 'c_v_sim': c_v_sim,
        }
