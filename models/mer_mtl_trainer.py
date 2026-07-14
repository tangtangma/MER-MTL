"""
MER-MTL Trainer
============================================================
Training logic for MER-MTL with DMD-aligned architecture.
Main task: 7-class CrossEntropy + Binary CrossEntropy (high weight: 1.0)
Auxiliary tasks: L_rec, L_cyc, L_mar, L_ort (uncertainty weighting)

Loss formula:
  L_total = L_main * 1.0 + sum(0.5/sigma_k^2 * L_aux_k) * aux_weight

Reference: Kendall et al. 2018 - "Multi-task learning using uncertainty to weigh losses"
============================================================
"""
import gc, logging, os, sys, time, math, argparse
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from scipy import stats

from models.mer_mtl_model import MERMTLModel


# =============================================================================
# METRICS (DMD-style for compatibility)
# =============================================================================
class MetricsTop:
    def __init__(self, train_mode):
        self.train_mode = train_mode

    def getMetics(self, dataset_name):
        return self._eval_regression

    def _multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def _eval_regression(self, y_pred, y_true):
        p = y_pred.view(-1).cpu().detach().numpy()
        t = y_true.view(-1).cpu().detach().numpy()
        mae = np.mean(np.absolute(p - t)).astype(np.float64)
        corr = np.corrcoef(p, t)[0][1]
        a7_p = np.clip(p, -3, 3)
        a7_t = np.clip(t, -3, 3)
        mult_a7 = self._multiclass_acc(a7_p, a7_t)
        nz = np.array([i for i, e in enumerate(t) if e != 0])
        if len(nz) == 0:
            acc2, f1 = 0.0, 0.0
        else:
            bt = (t[nz] > 0)
            bp = (p[nz] > 0)
            acc2 = float(np.mean(bp == bt))
            tp = np.sum((bp == 1) & (bt == 1))
            fp = np.sum((bp == 1) & (bt == 0))
            fn = np.sum((bp == 0) & (bt == 1))
            denom = 2 * tp + fp + fn
            f1 = float(2 * tp) / denom if denom > 0 else 0.0
        return {
            "Acc_2": round(acc2, 4),
            "F1_score": round(f1, 4),
            "Acc_7": round(float(mult_a7), 4),
            "MAE": round(float(mae), 4),
            "Corr": round(float(corr), 4) if not np.isnan(corr) else 0.0,
        }


# =============================================================================
# 7-CLASS + BINARY CLASSIFICATION METRICS (MER-MTL main task)
# =============================================================================
class Metrics7Class:
    """7-class + Binary classification metrics for MER-MTL evaluation"""
    
    def __init__(self):
        pass
    
    def compute(self, logits7, logits2, labels):
        """
        Args:
            logits7: (B, 7) raw logits for 7-class
            logits2: (B, 2) raw logits for binary
            labels: (B,) labels in -3..+3 range (float)
        Returns:
            dict with Acc_7, Acc_2, F1, MAE, Corr, etc.
            
            Metric alignment with DMD (run_dmd.py MetricsTop._eval_regression):
              - Acc_7: all samples (round then compare)
              - Acc_2: ONLY non-zero label samples (binary: >0 vs <=0)
              - F1_score: ONLY non-zero label samples (manual TP/FP/FN, same as DMD)
              - MAE: ALL samples (continuous regression MAE)
              - Corr: ALL samples (Pearson on all samples)
            
            Additional metrics (not in DMD, for reference only):
              - F1_weighted: 7-class weighted F1
              - MAE_class: class-distance MAE (0-6 range)
        """
        B = labels.size(0)
        
        # Convert to numpy for DMD-style computation
        p_cls = torch.argmax(logits7, dim=1)  # (B,) in 0..6
        p_cont = p_cls.float() - 3.0  # 0..6 -> -3..+3 (continuous mapping)
        preds2 = torch.argmax(logits2, dim=1)  # (B,) in {0, 1}
        
        p_np = p_cont.cpu().numpy()
        t_np = labels.cpu().numpy()
        pred2_np = preds2.cpu().numpy()
        targ2_np = (labels > 0).long().cpu().numpy()
        
        # --- Acc-7: ALL samples (round then compare, same as DMD) ---
        targets_cls = (labels.long() + 3).clamp(0, 6)  # (B,) in 0..6
        acc7 = (p_cls == targets_cls).float().mean().item()
        
        # --- MAE & Corr: ALL samples (same as DMD) ---
        mae = float(np.mean(np.abs(p_np - t_np)))
        c = np.corrcoef(p_np, t_np)[0, 1]
        corr = float(c) if not np.isnan(c) else 0.0
        
        # --- Acc-2 & F1: ONLY non-zero label samples (same as DMD) ---
        nz = np.array([i for i, e in enumerate(t_np) if e != 0])
        if len(nz) == 0:
            acc2, f1_bin = 0.0, 0.0
        else:
            # Binary: >0 positive, <=0 negative (same as DMD)
            bt = (t_np[nz] > 0)
            bp = (p_np[nz] > 0)  # use continuous-mapped prediction (same as DMD)
            
            # Acc-2 (non-zero only)
            acc2 = float(np.mean(bp == bt))
            
            # F1 (non-zero only, manual TP/FP/FN - exactly same as DMD)
            tp = np.sum((bp == 1) & (bt == 1))
            fp = np.sum((bp == 1) & (bt == 0))
            fn = np.sum((bp == 0) & (bt == 1))
            denom = 2 * tp + fp + fn
            f1_bin = float(2 * tp) / denom if denom > 0 else 0.0
        
        # --- Auxiliary metrics (not in DMD, for reference) ---
        # F1 weighted (7-class)
        from sklearn.metrics import f1_score
        pred_cls_np = p_cls.cpu().numpy()
        targ_cls_np = targets_cls.cpu().numpy()
        f1_w = f1_score(targ_cls_np, pred_cls_np, average='weighted', zero_division=0)
        
        # MAE class distance (0-6 range)
        mae_class = torch.abs(p_cls.float() - targets_cls.float()).mean().item()
        
        # Acc-2 breakdown (from 7-cls vs 2-cls head, for reference)
        preds2_from7 = (p_cls >= 4).long()
        targets2_all = (labels > 0).long()
        acc2_from7 = (preds2_from7 == targets2_all).float().mean().item()
        acc2_direct = (preds2 == targets2_all).float().mean().item()
        
        return {
            "Acc_7": round(float(acc7), 4),
            "Acc_2": round(float(acc2), 4),          # non-zero only (DMD-aligned)
            "Acc_2_from7": round(float(acc2_from7), 4),
            "Acc_2_direct": round(float(acc2_direct), 4),
            "F1_score": round(float(f1_bin), 4),      # non-zero only (DMD-aligned)
            "F1_binary": round(float(f1_bin), 4),      # alias
            "F1_weighted": round(float(f1_w), 4),      # 7-class weighted F1
            "MAE": round(float(mae), 4),               # all samples (DMD-aligned)
            "MAE_class": round(float(mae_class), 4),   # class-distance MAE
            "Corr": round(corr, 4),                     # all samples (DMD-aligned)
        }


# =============================================================================
# LOSS FUNCTIONS (same as DMD)
# =============================================================================
class MSE(nn.Module):
    """DMD-style MSE: sum over all elements / numel"""
    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        return torch.sum(diffs.pow(2)) / torch.numel(diffs.data)


class HingeLoss(nn.Module):
    """Verified against DMD official HingeLoss.py"""
    def __init__(self):
        super().__init__()

    def compute_cosine(self, x, y):
        x_n = torch.sqrt(torch.sum(torch.pow(x, 2), 1) + 1e-8)
        x_n = torch.max(x_n, 1e-8 * torch.ones_like(x_n))
        y_n = torch.sqrt(torch.sum(torch.pow(y, 2), 1) + 1e-8)
        y_n = torch.max(y_n, 1e-8 * torch.ones_like(y_n))
        return torch.sum(x * y, 1) / (x_n * y_n)

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape
        s = feats.repeat(1, B).view(-1, F)
        s_ids = ids.view(B, 1).repeat(1, B)
        t = feats.repeat(B, 1)
        t_ids = ids.view(1, B).repeat(B, 1)
        cosine = self.compute_cosine(s, t)
        eq = torch.eye(B, dtype=torch.bool, device=feats.device)
        s_ids = s_ids[~eq].view(B, B - 1)
        t_ids = t_ids[~eq].view(B, B - 1)
        cosine = cosine.view(B, B)[~eq].view(B, B - 1)
        sim_mask = (s_ids == t_ids)
        margin_val = 0.15 * torch.abs(s_ids - t_ids)
        loss = 0
        loss_num = 0
        for i in range(B):
            sn = sim_mask[i].sum().item()
            dn = B - 1 - sn
            if not sn or not dn:
                continue
            sc = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dn)
            dc = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sn).T
            tm = margin_val[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sn).T
            loss_i = torch.max(torch.zeros_like(sc), tm - sc + dc).mean()
            loss += loss_i
            loss_num += 1
        return loss / max(loss_num, 1)


# =============================================================================
# MER-MTL TRAINER (Main task high weight + Aux uncertainty weighting)
# =============================================================================
class MERMTLTrainer:
    """
    MER-MTL Trainer with:
      - Main task: 7-class CrossEntropy + Binary CrossEntropy (high weight: 1.0)
      - Auxiliary tasks: L_rec, L_cyc, L_mar, L_ort (uncertainty weighting)
    
    Loss formula:
      L_total = L_main * 1.0 + sum(0.5/sigma_k^2 * L_aux_k) * aux_weight
    
    Reference: Kendall et al. 2018 - "Multi-task learning using uncertainty to weigh losses"
    """
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Main task losses
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()
        
        # Auxiliary task losses (same as DMD)
        self.MSE = MSE()
        self.cosine = nn.CosineEmbeddingLoss(margin=0.1)
        self.sim_loss = HingeLoss()
        
        # Metrics
        self.metrics_top = MetricsTop(train_mode=args.train_mode)
        self.metrics_cls = Metrics7Class()
        
        # Uncertainty parameters (learnable log_sigmas for aux tasks)
        self.aux_task_names = ['L_rec', 'L_cyc', 'L_mar', 'L_ort']
        self.num_aux_tasks = len(self.aux_task_names)
        
        # Main task weight (fixed high weight)
        self.main_task_weight = 1.0
        # Auxiliary task weight (scaling factor)
        self.aux_task_weight = getattr(args, 'aux_task_weight', 0.1)
        
        # Initialize uncertainty parameters (create on device to keep leaf tensor)
        initial_log_sigma = getattr(args, 'initial_log_sigma', 0.0)
        self.log_sigmas = nn.Parameter(
            torch.full((self.num_aux_tasks,), float(initial_log_sigma), device=device),
            requires_grad=True)
        
        # Aligned/unaligned mode
        self.need_aligned = getattr(args, 'need_data_aligned', True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        self.logger = logging.getLogger('MMSA')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _prepare_batch(self, bd):
        """Prepare batch data with masks for aligned/unaligned mode"""
        T = bd['text'].to(self.device)
        A = bd['audio'].to(self.device)
        V = bd['vision'].to(self.device)
        L = bd['labels']['M'].to(self.device).squeeze(-1)  # (B,) in -3..+3
        
        # Check if batch has mask information
        text_mask = bd.get('text_mask', None)
        audio_mask = bd.get('audio_mask', None)
        vision_mask = bd.get('vision_mask', None)
        
        if text_mask is not None:
            text_mask = text_mask.to(self.device)
        if audio_mask is not None:
            audio_mask = audio_mask.to(self.device)
        if vision_mask is not None:
            vision_mask = vision_mask.to(self.device)
        
        return T, A, V, L, text_mask, audio_mask, vision_mask
    
    def _compute_aux_losses(self, out):
        """
        Compute 4 auxiliary losses (same as DMD official).
        Returns dict {L_rec, L_cyc, L_mar, L_ort}
        """
        # L_rec: Reconstruction loss (same as DMD)
        lr = self.MSE(out['recon_l'], out['origin_l'])
        lr += self.MSE(out['recon_a'], out['origin_a'])
        lr += self.MSE(out['recon_v'], out['origin_v'])
        
        # L_cyc: Cycle consistency (same as DMD)
        ls = self.MSE(out['s_l_seq'], out['s_l_r'])
        ls += self.MSE(out['s_a_seq'], out['s_a_r'])
        ls += self.MSE(out['s_v_seq'], out['s_v_r'])
        
        # L_ort: Orthogonality on s/c vectors (same as DMD)
        B = out['c_L_raw'].size(0)
        neg = torch.full((B,), -1.0, device=self.device)
        lo = self.cosine(out['s_L_raw'], out['c_L_raw'], neg)
        lo += self.cosine(out['s_A_raw'], out['c_A_raw'], neg)
        lo += self.cosine(out['s_V_raw'], out['c_V_raw'], neg)
        
        # L_mar: Margin ranking (same as DMD) - use raw content features
        fl, il = [], []
        reg_labels = out.get('labels_for_margin', None)
        if reg_labels is None:
            reg_labels = out['emotion_logits'].mean(dim=1)
        for i in range(B):
            fl.extend([out['c_L_raw'][i:i+1], out['c_A_raw'][i:i+1], out['c_V_raw'][i:i+1]])
            il.extend([reg_labels[i:i+1]] * 3)
        feats = torch.cat(fl, dim=0)
        ids = torch.cat(il, dim=0)
        lsim = self.sim_loss(ids, feats)
        
        return {
            'L_rec': lr,
            'L_cyc': ls,
            'L_mar': lsim,
            'L_ort': lo,
        }
    
    def _compute_main_loss(self, out, labels):
        """
        Main task: 7-class CrossEntropy + Binary CrossEntropy.
        Labels: (B,) in range [-3, +3] -> convert to [0, 6] for 7-class, [0, 1] for binary
        """
        # 7-class CrossEntropy
        logits_7 = out['emotion_logits']  # (B, 7)
        targets_7 = (labels.long() + 3).clamp(0, 6)  # (B,) in 0..6
        loss_7cls = self.criterion_cls(logits_7, targets_7)
        
        # Binary CrossEntropy
        logits_2 = out['emotion_logits_2cls']  # (B, 2)
        targets_2 = (labels > 0).float()  # (B,) in {0, 1}
        loss_2cls = self.criterion_bce(logits_2[:, 1] - logits_2[:, 0], targets_2)
        
        # Combined main task loss
        loss_main = loss_7cls + 0.5 * loss_2cls
        return loss_main
    
    def _uncertainty_weighted_aux(self, aux_losses):
        """
        Kendall 2018 uncertainty weighting on auxiliary losses.
        
        Formula: sum(0.5 / sigma^2 * L_k)
        NOTE: NO +log(sigma) term (that causes loss to go negative)
        """
        loss_values = []
        for key in self.aux_task_names:
            v = aux_losses.get(key, torch.tensor(0.0, device=self.device))
            if isinstance(v, torch.Tensor):
                loss_values.append(F.relu(v))
            else:
                loss_values.append(torch.tensor(max(0.0, v), device=self.device))
        
        loss_tensor = torch.stack(loss_values)
        eps = 1e-3
        sigmas_sq = torch.exp(2 * self.log_sigmas[:self.num_aux_tasks]) + eps
        precision_weights = 1.0 / (2 * sigmas_sq)
        
        # CORRECT: just weighted sum, NO +log(sigma)
        weighted_aux = (precision_weights * loss_tensor).sum()
        return weighted_aux
    
    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model.parameters()) + [self.log_sigmas]
        opt = optim.Adam(params, lr=self.args.learning_rate,
                         weight_decay=self.args.weight_decay)
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, 
                                  patience=self.args.patience)
        
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        
        best_valid_acc7 = 0
        pt_dir = Path(getattr(self.args, 'checkpoint_dir', './checkpoints/mermtl'))
        pt_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            epochs += 1
            model.train()
            
            train_loss = 0.0
            train_loss_main = 0.0
            train_loss_aux = 0.0
            
            y_pred_logits_7, y_pred_logits_2, y_true = [], [], []
            
            left = getattr(self.args, 'update_epochs', 10)
            
            with tqdm(dataloader['train'], desc=f"Ep {epochs}", ascii=False, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as td:
                for batch_idx, bd in enumerate(td):
                    if left == getattr(self.args, 'update_epochs', 10):
                        opt.zero_grad()
                    left -= 1
                    
                    T, A, V, L, text_mask, audio_mask, vision_mask = self._prepare_batch(bd)
                    
                    out = model(T, A, V, text_mask, audio_mask, vision_mask, is_distill=True)
                    
                    # Add labels for margin loss
                    out['labels_for_margin'] = L
                    
                    # Main task: 7-class + Binary CrossEntropy (HIGH WEIGHT: 1.0)
                    loss_main = self._compute_main_loss(out, L)
                    
                    # Auxiliary losses (same as DMD)
                    aux_losses = self._compute_aux_losses(out)
                    
                    # Uncertainty weighting on auxiliary losses
                    if getattr(self.args, 'use_uncertainty', True):
                        weighted_aux = self._uncertainty_weighted_aux(aux_losses)
                    else:
                        weighted_aux = sum(v for v in aux_losses.values())
                    
                    # Total loss: main (weight=1.0) + weighted aux (weight=0.1)
                    total = loss_main * self.main_task_weight + weighted_aux * self.aux_task_weight
                    
                    total.backward()
                    if getattr(self.args, 'grad_clip', -1.0) != -1.0:
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)
                    
                    train_loss += total.item()
                    train_loss_main += loss_main.item()
                    train_loss_aux += weighted_aux.item()
                    
                    y_pred_logits_7.append(out['emotion_logits'].cpu())
                    y_pred_logits_2.append(out['emotion_logits_2cls'].cpu())
                    y_true.append(L.cpu())
                    
                    # Update progress bar with current loss
                    avg_loss = train_loss / (batch_idx + 1)
                    td.set_postfix({'loss': f'{avg_loss:.4f}', 'main': f'{train_loss_main/(batch_idx+1):.4f}'})
                    
                    if not left:
                        opt.step()
                        left = getattr(self.args, 'update_epochs', 10)
            
            if not left:
                opt.step()
            
            n_batches = len(dataloader['train'])
            train_loss /= n_batches
            train_loss_main /= n_batches
            train_loss_aux /= n_batches
            
            # Compute train metrics
            pred_7 = torch.cat(y_pred_logits_7)
            pred_2 = torch.cat(y_pred_logits_2)
            true_cls = torch.cat(y_true)
            tr_metrics = self.metrics_cls.compute(pred_7, pred_2, true_cls)
            tr_metrics["Loss"] = round(train_loss, 4)
            
            self.logger.info(f"TRAIN >> Acc7={tr_metrics['Acc_7']:.2%} | "
                           f"Acc2={tr_metrics['Acc_2']:.2%} | "
                           f"F1={tr_metrics['F1_score']:.4f} | "
                           f"Loss={train_loss:.4f} (main={train_loss_main:.4f}, aux={train_loss_aux:.4f})")
            
            # Validation & Test
            vr = self.do_test(model, dataloader['valid'], mode="VAL")
            ter = self.do_test(model, dataloader['test'], mode="TEST")
            
            cur_v = vr['Acc_7']
            sched.step(train_loss)
            
            torch.save(model.state_dict(), pt_dir / f"{epochs}.pth")
            
            better = cur_v >= best_valid_acc7 + 1e-6
            if better:
                best_valid_acc7, best_epoch = cur_v, epochs
                torch.save(model.state_dict(), pt_dir / self.args.save_name)
            
            sigmas_str = ', '.join([f"{s:.3f}" for s in torch.exp(self.log_sigmas).tolist()])
            
            self.logger.info(
                f"Ep {epochs}/{epochs-best_epoch} | Best:{best_epoch} | "
                f"VAL Acc7={vr['Acc_7']:.2%} Acc2={vr['Acc_2']:.2%} | "
                f"TEST Acc7={ter['Acc_7']:.2%} | "
                f"sigmas=[{sigmas_str}]")
            
            if return_epoch_results:
                epoch_results['train'].append(tr_metrics)
                epoch_results['valid'].append(vr)
                epoch_results['test'].append(ter)
            
            if epochs - best_epoch >= getattr(self.args, 'early_stop', 15):
                return epoch_results if return_epoch_results else None
    
    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        
        y_pred_logits_7, y_pred_logits_2, y_true = [], [], []
        
        with torch.no_grad():
            with tqdm(dataloader, desc=mode) as td:
                for bd in td:
                    T, A, V, L, text_mask, audio_mask, vision_mask = self._prepare_batch(bd)
                    
                    out = model(T, A, V, text_mask, audio_mask, vision_mask, is_distill=True)
                    
                    y_pred_logits_7.append(out['emotion_logits'].cpu())
                    y_pred_logits_2.append(out['emotion_logits_2cls'].cpu())
                    y_true.append(L.cpu())
        
        pred_7 = torch.cat(y_pred_logits_7)
        pred_2 = torch.cat(y_pred_logits_2)
        true_cls = torch.cat(y_true)
        metrics = self.metrics_cls.compute(pred_7, pred_2, true_cls)
        
        self.logger.info(f"{mode} >> Acc7={metrics['Acc_7']:.2%} | "
                        f"Acc2={metrics['Acc_2']:.2%} | "
                        f"F1={metrics['F1_score']:.4f}")
        return metrics


# =============================================================================
# CONFIGURATION (supports both aligned and unaligned)
# =============================================================================
def get_args(mode='aligned', cls_mode='7cls', epochs=30, batch=16, lr=0.0001,
            seeds=[42], aux_weight=0.1, text_mode='tt',
            checkpoint_dir='./pt/mermtl'):
    """
    Build args Namespace compatible with MERMTLModel + MERMTLTrainer.
    
    Key settings:
      - DMD architecture: dst=50, nheads=10, nlevels=4
      - 7-class classification + Binary classification as main task
      - 4 auxiliary tasks aligned with DMD
      - Uncertainty weighting on aux tasks
      - Main task weight: 1.0 (high), Auxiliary weight: 0.1 (via uncertainty)
      - Supports both ALIGNED and UNALIGNED data with attention masking
      - text_mode: 'tt' (Text Transformer) or 'mp' (Mean Pooling)
    
    Mode options:
      - 'aligned': uses aligned_50.pkl, masks are None (all modalities time-synced)
      - 'unaligned': uses unaligned_50.pkl, may have per-modality masks
    
    Text mode options:
      - 'tt': Conv1d -> Transformer (temporal modeling for text)
      - 'mp': Conv1d -> MeanPool -> MLP (no temporal modeling for text)
    """
    aligned = (mode == 'aligned')
    return Namespace(
        dataset_name='mosi',
        model_name='mer-mtl',
        featurePath=f"./data/{mode}_50.pkl",
        train_mode='regression',
        KeyEval='Acc_7',
        need_data_aligned=aligned,
        text_mode=text_mode,  # 'tt' or 'mp'
        checkpoint_dir=checkpoint_dir,
        feature_dims=[768, 5, 20],
        dst_feature_dim_nheads=[50, 10],
        nlevels=4,
        conv1d_kernel_size_l=5,
        conv1d_kernel_size_a=5,
        conv1d_kernel_size_v=5,
        attn_dropout=0.3,
        text_dropout=0.5,
        output_dropout=0.5,
        embed_dropout=0.2,
        relu_dropout=0.0,
        res_dropout=0.0,
        learning_rate=lr,
        grad_clip=0.6,
        patience=5,
        weight_decay=0.0,
        update_epochs=10,
        early_stop=15,
        batch_size=batch,
        attn_mask=False,
        use_uncertainty=True,
        initial_log_sigma=0.0,
        aux_task_weight=aux_weight,
        epochs=epochs,
        seeds=seeds,
        save_name=f'mer_mtl_{mode}_{text_mode}_{cls_mode}.pth',
    )
