"""
DMD Baseline Training Script

Reproducing the DMD model from CVPR 2023 paper:
Decoupled Multimodal Distilling for Emotion Recognition.

The model architecture is an exact implementation of
trains/singleTask/model/dmd.py + DMD.py.

Hyperparameters are adjusted to the MMSA-aligned 50-dim feature distribution
(data/aligned_50.pkl).

All architectural components (Conv1d projections, TransformerEncoder,
DistillationKernels, loss functions) are faithfully reproduced from the
official DMD codebase.

Usage:
    Aligned:   python run_dmd.py aligned --multi_seed --epochs 30
    Unaligned: python run_dmd.py unaligned --multi_seed --epochs 30
    Both:      python run_dmd.py all --multi_seed --epochs 30
"""

import gc, json, logging, os, sys, time, math
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
from scipy import stats

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


# =============================================================================
# PART 1: METRICS
# =============================================================================

class MetricsTop:
    def __init__(self, train_mode):
        self.train_mode = train_mode

    def getMetics(self, dataset_name):
        if self.train_mode == "regression":
            return {'MOSI': self._eval_regression, 'MOSEI': self._eval_regression}.get(
                dataset_name.upper(), self._eval_regression)
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
# PART 2: TRANSFORMER ENCODER (from trains/subNets/transformers_encoder/)
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
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
            self.weights[device] = self.weights[device].type_as(self._float_tensor).to(input.device)
        positions = self._make_positions(input)
        return self.weights[device].index_select(0, positions.contiguous().view(-1)).view(bsz, seq_len, -1).detach()

    def _make_positions(self, tensor):
        max_pos = self.padding_idx + 1 + tensor.size(1)
        device = tensor.get_device()
        buf_name = f'rp_{device}'
        if not hasattr(self, buf_name):
            setattr(self, buf_name, tensor.new())
        buf = getattr(self, buf_name)
        if buf.numel() < max_pos:
            new_buf = tensor.new_empty(max_pos)
            setattr(self, buf_name, new_buf)
            buf = new_buf
        torch.arange(self.padding_idx + 1, max_pos, out=buf)
        mask = tensor.ne(self.padding_idx)
        positions = buf[:tensor.size(1)].expand_as(tensor)
        if self.left_pad:
            positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
            new_t = tensor.clone()
            return new_t.masked_scatter_(mask, positions[mask]).long()
        return tensor.ne(self.padding_idx).long().cumsum(dim=1) + self.padding_idx


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, edim = query.size()

        if qkv_same:
            q, k, v = self._qkv(query)
        elif kv_same:
            q = self._q(query)
            k, v = self._kv(key)
        else:
            q = self._q(query)
            k = self._k(key)
            v = self._v(value)

        q = q * self.scaling

        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_w = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_w = attn_w + attn_mask
        attn_w = F.softmax(attn_w.float(), dim=-1).type_as(attn_w)
        attn_w = F.dropout(attn_w, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_w, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, edim)
        attn = self.out_proj(attn)
        attn_w = attn_w.view(bsz, self.num_heads, tgt_len, src_len).sum(dim=1) / self.num_heads

        return attn, attn_w

    def _qkv(self, x):
        return self._in_proj(x).chunk(3, dim=-1)

    def _kv(self, x):
        return self._in_proj(x, start=self.embed_dim).chunk(2, dim=-1)

    def _q(self, x, **kw):
        return self._in_proj(x, end=self.embed_dim, **kw)

    def _k(self, x):
        return self._in_proj(x, start=self.embed_dim, end=2 * self.embed_dim)

    def _v(self, x):
        return self._in_proj(x, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kw):
        w = kw.get('weight', self.in_proj_weight)[start:end, :]
        b = kw.get('bias', self.in_proj_bias)
        if b is not None:
            b = b[start:end]
        return F.linear(input, w, b)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self._ln(0, x, before=True)
        mask = self._fut_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self._ln(0, x_k, before=True)
            x_v = self._ln(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._ln(0, x, after=True)

        residual = x
        x = self._ln(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._ln(1, x, after=True)

        return x

    def _ln(self, layer_idx, x, before=False, after=False):
        assert before ^ after
        li_is_data = hasattr(layer_idx, 'ndim') and layer_idx.ndim >= 2
        x_is_int = isinstance(x, int)
        if li_is_data and x_is_int:
            layer_idx, x = x, layer_idx
        if isinstance(layer_idx, int):
            idx = layer_idx
        else:
            raise TypeError(f"_ln expects int as 1st arg, got {type(layer_idx).__name__} shape={getattr(layer_idx,'shape',None)}")
        return self.layer_norms[idx](x) if after ^ self.normalize_before else x

    def _fut_mask(self, t, t_k=None):
        tgt_len = t.size(0)
        src_len = tgt_len if t_k is None else t_k.size(0)
        m = torch.triu(
            torch.ones(tgt_len, src_len).float().fill_(float('-inf')).type_as(t),
            1)
        return m


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, num_heads=num_heads, attn_dropout=attn_dropout,
                relu_dropout=relu_dropout, res_dropout=res_dropout, attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        x = self.embed_scale * x_in
        # FIX: use non-inplace addition to avoid modifying x_in's data buffer
        x = x + self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            x_k = x_k + self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_v = x_v + self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, x_k if x_in_k is not None else None, x_v if x_in_v is not None else None)

        if self.normalize:
            x = self.layer_norm(x)
        return x


# =============================================================================
# PART 3: LOSS FUNCTIONS (from trains/singleTask/)
# =============================================================================

class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        return torch.sum(diffs.pow(2)) / torch.numel(diffs.data)


class HingeLoss(nn.Module):
    """
    Verified against trains/singleTask/HingeLoss.py.
    margin is always overridden to 0.15*abs(s_ids-t_ids) inside forward().
    """
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
            # FIX: use non-inplace addition for autograd compatibility
            loss = loss + loss_i
            loss_num += 1

        return loss / max(loss_num, 1)


def min_cosine(student, teacher, weights=None):
    fn = nn.CosineEmbeddingLoss(reduction='none')
    target = torch.full((student.size(0),), -1.0, device=student.device)
    dists = fn(student, teacher.detach(), target)
    return dists.mean() if weights is None else (dists * weights).mean()


def distance_metric(student, teacher, option, weights=None):
    if option == 'cosine':
        dists = 1 - F.cosine_similarity(student, teacher.detach(), dim=1)
    elif option == 'l2':
        dists = (student - teacher.detach()).pow(2).sum(1)
    elif option == 'l1':
        dists = torch.abs(student - teacher.detach()).sum(1)
    elif option == 'kl':
        T = 8
        dists = F.kl_div(F.log_softmax(student / T), F.softmax(teacher.detach() / T)) * (T * T)
    else:
        raise NotImplementedError(f"Unknown metric: {option}")
    return dists.mean() if weights is None else (dists * weights).mean()


def softmax(w, t=1.0, axis=None):
    w = np.array(w, dtype=np.float32) / t
    e = np.exp(w - np.amax(w, axis=axis, keepdims=True))
    dist = e / np.sum(e, axis=axis, keepdims=True)
    return dist


# =============================================================================
# PART 4: DMD MODEL (exact match to trains/singleTask/model/dmd.py)
# =============================================================================

class DMDModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        dst_dim, nheads = args.dst_feature_dim_nheads
        self.d = dst_dim
        self.nh = nheads
        self.layers = args.nlevels

        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = getattr(args, 'attn_dropout_a', args.attn_dropout)
        self.attn_dropout_v = getattr(args, 'attn_dropout_v', args.attn_dropout)
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        ks_l = args.conv1d_kernel_size_l
        ks_a = args.conv1d_kernel_size_a
        ks_v = args.conv1d_kernel_size_v
        self.ks_l, self.ks_a, self.ks_v = ks_l, ks_a, ks_v

        self.need_aligned = getattr(args, 'need_data_aligned', True)
        if self.need_aligned:
            self.seq_l = 50
            self.seq_v = 50
            self.seq_a = 50
        else:
            self.seq_l = 50
            self.seq_a = 375
            self.seq_v = 500

        self.eff_l = self.seq_l - ks_l + 1
        self.eff_v = self.seq_v - ks_v + 1
        self.eff_a = self.seq_a - ks_a + 1

        d_l, d_a, d_v = args.feature_dims
        cL = self.d
        cH = 2 * self.d
        cF = 3 * self.d
        cTot = 2 * (self.d + self.d + self.d) + 3 * self.d

        # 1. Conv1d projections
        self.proj_l = nn.Conv1d(d_l, self.d, kernel_size=ks_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(d_a, self.d, kernel_size=ks_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(d_v, self.d, kernel_size=ks_v, padding=0, bias=False)

        # 2. Specific encoders
        self.enc_s_l = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.enc_s_v = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.enc_s_a = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
        self.enc_c = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)

        # 3. Decoders
        self.dec_l = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        self.dec_v = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)
        self.dec_a = nn.Conv1d(self.d * 2, self.d, kernel_size=1, padding=0, bias=False)

        # 4. Cosine sim projections
        self.proj_cos_l = nn.Linear(cL * self.eff_l, cL)
        self.proj_cos_v = nn.Linear(cL * self.eff_v, cL)
        self.proj_cos_a = nn.Linear(cL * self.eff_a, cL)

        # 5. Alignment projections
        self.align_l = nn.Linear(cL * self.eff_l, cL)
        self.align_v = nn.Linear(cL * self.eff_v, cL)
        self.align_a = nn.Linear(cL * self.eff_a, cL)

        # 6. Self-attention for c vectors
        self.attn_c_l = self._mk_attn(self.d, self.attn_dropout)
        self.attn_c_v = self._mk_attn(self.d, self.attn_dropout_v)
        self.attn_c_a = self._mk_attn(self.d, self.attn_dropout_a)

        # 7. Fusion projection
        self.proj1_c = nn.Linear(cF, cF)
        self.proj2_c = nn.Linear(cF, cF)
        self.out_c = nn.Linear(cF, 1)

        # 8. Homogeneous GD
        self.proj1_ll = nn.Linear(cL * self.eff_l, cL)
        self.proj2_ll = nn.Linear(cL, cL * self.eff_l)
        self.out_ll = nn.Linear(cL * self.eff_l, 1)

        self.proj1_lv = nn.Linear(cL * self.eff_v, cL)
        self.proj2_lv = nn.Linear(cL, cL * self.eff_v)
        self.out_lv = nn.Linear(cL * self.eff_v, 1)

        self.proj1_la = nn.Linear(cL * self.eff_a, cL)
        self.proj2_la = nn.Linear(cL, cL * self.eff_a)
        self.out_la = nn.Linear(cL * self.eff_a, 1)

        # 9. Heterogeneous GD
        self.proj1_lh = nn.Linear(cH, cH)
        self.proj2_lh = nn.Linear(cH, cH)
        self.out_lh = nn.Linear(cH, 1)

        self.proj1_vh = nn.Linear(cH, cH)
        self.proj2_vh = nn.Linear(cH, cH)
        self.out_vh = nn.Linear(cH, 1)

        self.proj1_ah = nn.Linear(cH, cH)
        self.proj2_ah = nn.Linear(cH, cH)
        self.out_ah = nn.Linear(cH, 1)

        # 10. Ensemble
        self.w_l = nn.Linear(2 * self.d, 2 * self.d)
        self.w_v = nn.Linear(2 * self.d, 2 * self.d)
        self.w_a = nn.Linear(2 * self.d, 2 * self.d)
        self.w_c = nn.Linear(3 * self.d, 3 * self.d)
        self.proj1 = nn.Linear(cTot, cTot)
        self.proj2 = nn.Linear(cTot, cTot)
        self.out_layer = nn.Linear(cTot, 1)

        # 11. Cross-modal attention (6 paths)
        self.t_l_la = self._mk_attn(self.d, self.attn_dropout)
        self.t_l_lv = self._mk_attn(self.d, self.attn_dropout)

        self.t_a_al = self._mk_attn(self.d, self.attn_dropout_a)
        self.t_a_av = self._mk_attn(self.d, self.attn_dropout_a)

        self.t_v_vl = self._mk_attn(self.d, self.attn_dropout_v)
        self.t_v_va = self._mk_attn(self.d, self.attn_dropout_v)

        self.t_l_mem = self._mk_attn(self.d * 2, self.attn_dropout, layers=3)
        self.t_a_mem = self._mk_attn(self.d * 2, self.attn_dropout, layers=3)
        # Official DMD uses attn_dropout=0.3 for t_a_mem (same as text), not attn_dropout_a=0.2
        self.t_v_mem = self._mk_attn(self.d * 2, self.attn_dropout_v, layers=3)

        # 12. Output bias zeroing to eliminate initial positive bias
        # Only zero task output layers, not internal layers (to preserve gradient flow)
        self.out_layer.bias.data.zero_()
        self.out_c.bias.data.zero_()
        self.out_ll.bias.data.zero_()
        self.out_lv.bias.data.zero_()
        self.out_la.bias.data.zero_()
        self.out_lh.bias.data.zero_()
        self.out_vh.bias.data.zero_()
        self.out_ah.bias.data.zero_()

    def _mk_attn(self, edim, adrop, layers=None):
        return TransformerEncoder(
            edim, num_heads=self.nh,
            layers=max(self.layers, layers) if layers else self.layers,
            attn_dropout=adrop, relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout, embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask)

    def forward(self, text, audio, video, is_distill=False):
        B = text.size(0)

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        px_l = self.proj_l(x_l)
        px_a = self.proj_a(x_a)
        px_v = self.proj_v(x_v)

        s_l = self.enc_s_l(px_l)
        s_v = self.enc_s_v(px_v)
        s_a = self.enc_s_a(px_a)

        c_l = self.enc_c(px_l)
        c_v = self.enc_c(px_v)
        c_a = self.enc_c(px_a)

        c_l_s = self.align_l(c_l.contiguous().view(B, -1))
        c_v_s = self.align_v(c_v.contiguous().view(B, -1))
        c_a_s = self.align_a(c_a.contiguous().view(B, -1))

        r_l = self.dec_l(torch.cat([s_l, c_l], dim=1))
        r_v = self.dec_v(torch.cat([s_v, c_v], dim=1))
        r_a = self.dec_a(torch.cat([s_a, c_a], dim=1))

        s_l_r = self.enc_s_l(r_l)
        s_v_r = self.enc_s_v(r_v)
        s_a_r = self.enc_s_a(r_a)

        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        # Homogeneous GD
        # FIX: use .clone() to detach view from shared memory before using as residual
        h_ll = c_l.transpose(0, 1).contiguous().view(B, -1).clone()
        r_ll = self.proj1_ll(h_ll)
        # FIX: use non-inplace relu to avoid modifying autograd graph
        h_ll_p = self.proj2_ll(F.dropout(F.relu(r_ll), p=self.output_dropout, training=self.training))
        h_ll_p = h_ll_p + h_ll
        l_ll = self.out_ll(h_ll_p)

        h_lv = c_v.transpose(0, 1).contiguous().view(B, -1).clone()
        r_lv = self.proj1_lv(h_lv)
        h_lv_p = self.proj2_lv(F.dropout(F.relu(r_lv), p=self.output_dropout, training=self.training))
        h_lv_p = h_lv_p + h_lv
        l_lv = self.out_lv(h_lv_p)

        h_la = c_a.transpose(0, 1).contiguous().view(B, -1).clone()
        r_la = self.proj1_la(h_la)
        h_la_p = self.proj2_la(F.dropout(F.relu(r_la), p=self.output_dropout, training=self.training))
        h_la_p = h_la_p + h_la
        l_la = self.out_la(h_la_p)

        # Cosine sim projections
        ps_l = self.proj_cos_l(s_l.transpose(0, 1).contiguous().view(B, -1))
        ps_v = self.proj_cos_v(s_v.transpose(0, 1).contiguous().view(B, -1))
        ps_a = self.proj_cos_a(s_a.transpose(0, 1).contiguous().view(B, -1))

        # Self-attention on c
        c_l_a = self.attn_c_l(c_l)[-1]
        c_v_a = self.attn_c_v(c_v)[-1]
        c_a_a = self.attn_c_a(c_a)[-1]

        # Fusion: concat features along feature dim -> (B, 3d)
        # Note: [-1] above already extracted last timestep, so shape is (B, d)
        c_f = torch.cat([c_l_a, c_v_a, c_a_a], dim=1)
        c_p = self.proj2_c(F.dropout(F.relu(self.proj1_c(c_f)), p=self.output_dropout, training=self.training))
        c_p = c_p + c_f
        l_c = self.out_c(c_p)

        # Cross-modal attention
        h_la2 = self.t_l_la(s_l, s_a, s_a)
        h_lv2 = self.t_l_lv(s_l, s_v, s_v)
        h_ls = torch.cat([h_la2, h_lv2], dim=2)
        h_ls = self.t_l_mem(h_ls)[-1]

        h_al = self.t_a_al(s_a, s_l, s_l)
        h_av = self.t_a_av(s_a, s_v, s_v)
        h_as = torch.cat([h_al, h_av], dim=2)
        h_as = self.t_a_mem(h_as)[-1]

        h_vl = self.t_v_vl(s_v, s_l, s_l)
        h_va = self.t_v_va(s_v, s_a, s_a)
        h_vs = torch.cat([h_vl, h_va], dim=2)
        h_vs = self.t_v_mem(h_vs)[-1]

        # Note: [-1] above already extracted last timestep, so h_ls/h_vs/h_as are (B, 2d)

        # Heterogeneous GD (operates on batch-level features)
        h_lh = self.proj2_lh(F.dropout(F.relu(self.proj1_lh(h_ls)), p=self.output_dropout, training=self.training))
        h_lh = h_lh + h_ls
        l_lh = self.out_lh(h_lh)

        h_vh = self.proj2_vh(F.dropout(F.relu(self.proj1_vh(h_vs)), p=self.output_dropout, training=self.training))
        h_vh = h_vh + h_vs
        l_vh = self.out_vh(h_vh)

        h_ah = self.proj2_ah(F.dropout(F.relu(self.proj1_ah(h_as)), p=self.output_dropout, training=self.training))
        h_ah = h_ah + h_as
        l_ah = self.out_ah(h_ah)

        # Ensemble: concat along feature dim -> (B, 9d) = (B, 450)
        wl = torch.sigmoid(self.w_l(h_ls))   # (B, 2d)
        wv = torch.sigmoid(self.w_v(h_vs))   # (B, 2d)
        wa = torch.sigmoid(self.w_a(h_as))   # (B, 2d)
        wc = torch.sigmoid(self.w_c(c_f))            # (B, 3d)
        ens = torch.cat([wl, wv, wa, wc], dim=1)     # (B, 9d=450)
        ens_p = self.proj2(F.dropout(F.relu(self.proj1(ens)), p=self.output_dropout, training=self.training))
        ens_p = ens_p + ens
        out = self.out_layer(ens_p)                   # (B, 1)

        return {
            'logits_l_homo': l_ll, 'logits_v_homo': l_lv, 'logits_a_homo': l_la,
            'repr_l_homo': r_ll, 'repr_v_homo': r_lv, 'repr_a_homo': r_la,
            'origin_l': px_l, 'origin_v': px_v, 'origin_a': px_a,
            's_l': s_l, 's_v': s_v, 's_a': s_a,
            'proj_s_l': ps_l, 'proj_s_v': ps_v, 'proj_s_a': ps_a,
            'c_l': c_l, 'c_v': c_v, 'c_a': c_a,
            's_l_r': s_l_r, 's_v_r': s_v_r, 's_a_r': s_a_r,
            'recon_l': r_l, 'recon_v': r_v, 'recon_a': r_a,
            'c_l_sim': c_l_s, 'c_v_sim': c_v_s, 'c_a_sim': c_a_s,
            'logits_l_hetero': l_lh, 'logits_v_hetero': l_vh, 'logits_a_hetero': l_ah,
            'repr_l_hetero': h_lh, 'repr_v_hetero': h_vh, 'repr_a_hetero': h_ah,
            'last_h_l': h_ls, 'last_h_v': h_vs, 'last_h_a': h_as,
            'logits_c': l_c,
            'output_logit': out,
        }


# =============================================================================
# PART 5: DISTILLATION KERNELS (exact match to trains/singleTask/distillnets/)
# =============================================================================

class DistillationKernelHomo(nn.Module):
    def __init__(self, n_classes, hidden_size, gd_size, to_idx, from_idx, gd_prior, gd_reg, w_losses, metric, alpha):
        super().__init__()
        self.W_logit = nn.Linear(n_classes, gd_size)
        self.W_repr = nn.Linear(hidden_size, gd_size)
        self.W_edge = nn.Linear(gd_size * 4, 1)
        self.gd_size = gd_size
        self.to_idx = to_idx
        self.from_idx = from_idx
        self.alpha = alpha
        self.register_buffer('_gp', torch.FloatTensor(gd_prior))
        self.gd_reg = gd_reg
        self.w_losses = w_losses
        self.metric = metric

    def forward(self, logits, reprs):
        nm, bs = logits.size()[:2]
        z_l = self.W_logit(logits.view(nm * bs, -1))
        z_r = self.W_repr(reprs.view(nm * bs, -1))
        z = torch.cat([z_l, z_r], dim=1).view(nm, bs, self.gd_size * 2)

        edges = []
        for j in self.to_idx:
            for i in self.from_idx:
                if i != j:
                    edges.append(self.W_edge(torch.cat([z[j], z[i]], dim=1)))
        edges = torch.cat(edges, dim=1)
        e_orig = edges.sum(0).unsqueeze(0).transpose(0, 1)
        edges = F.softmax(edges * self.alpha, dim=1).transpose(0, 1)
        return edges, e_orig

    def distillation_loss(self, logits, reprs, edges):
        gp = self._gp.to(edges.device)
        loss_reg = (edges.mean(1) - gp).pow(2).sum() * self.gd_reg
        loss_logit, loss_repr = 0.0, 0.0
        x = 0
        for j in self.to_idx:
            for i, idx in enumerate(self.from_idx):
                if i != j:
                    wd = edges[x] + gp[x]
                    # FIX: use non-inplace addition for autograd compatibility
                    loss_logit = loss_logit + self.w_losses[0] * distance_metric(logits[j], logits[idx], self.metric, wd)
                    loss_repr = loss_repr + self.w_losses[1] * distance_metric(reprs[j], reprs[idx], self.metric, wd)
                    x += 1
        return loss_reg, loss_logit, loss_repr


class DistillationKernelHetero(nn.Module):
    def __init__(self, n_classes, hidden_size, gd_size, to_idx, from_idx, gd_prior, gd_reg, w_losses, metric, alpha):
        super().__init__()
        self.W_logit = nn.Linear(n_classes, gd_size)
        self.W_repr = nn.Linear(hidden_size, gd_size)
        self.W_edge = nn.Linear(gd_size * 4, 1)
        self.gd_size = gd_size
        self.to_idx = to_idx
        self.from_idx = from_idx
        self.alpha = alpha
        self.register_buffer('_gp', torch.FloatTensor(gd_prior))
        self.gd_reg = gd_reg
        self.w_losses = w_losses
        self.metric = metric

    def forward(self, logits, reprs):
        nm, bs = logits.size()[:2]
        z_l = self.W_logit(logits.view(nm * bs, -1))
        z_r = self.W_repr(reprs.view(nm * bs, -1))
        z_l_expanded = z_l.expand(-1, self.gd_size)
        z = torch.cat([z_l_expanded, z_r], dim=1).view(nm, bs, self.gd_size * 2)

        edges = []
        for j in self.to_idx:
            for i in self.from_idx:
                if i != j:
                    edges.append(self.W_edge(torch.cat([z[j], z[i]], dim=1)))
        edges = torch.cat(edges, dim=1)
        e_orig = edges.sum(0).unsqueeze(0).transpose(0, 1)
        edges = F.softmax(edges * self.alpha, dim=1).transpose(0, 1)
        return edges, e_orig

    def distillation_loss(self, logits, reprs, edges):
        gp = self._gp.to(edges.device)
        loss_reg = (edges.mean(1) - gp).pow(2).sum() * self.gd_reg
        loss_logit, loss_repr = 0.0, 0.0
        x = 0
        for j in self.to_idx:
            for i, idx in enumerate(self.from_idx):
                if i != j:
                    wd = edges[x] + gp[x]
                    # FIX: use non-inplace addition for autograd compatibility
                    loss_logit = loss_logit + self.w_losses[0] * distance_metric(logits[j], logits[idx], self.metric, wd)
                    rj = reprs[j].view(-1, reprs[j].size(-1))
                    ri = reprs[idx].view(-1, reprs[idx].size(-1))
                    loss_repr = loss_repr + self.w_losses[1] * min_cosine(rj, ri, wd)
                    x += 1
        return loss_reg, loss_logit, loss_repr


# =============================================================================
# PART 6: DMD TRAINER (exact match to trains/singleTask/DMD.py)
# =============================================================================

class DMDTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def _str(self, r):
        return " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in r.items())

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = (list(model[0].parameters()) + list(model[1].parameters()) + list(model[2].parameters()))
        opt = optim.Adam(params, lr=self.args.learning_rate)
        # Official DMD train.py: Adam without weight_decay, ReduceLROnPlateau without verbose
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        minmax = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if minmax == 'min' else 0
        mdl, homo, hete = model

        # FIX: output path corrected to ./pt/dmd/
        pt_dir = Path("./pt/dmd")
        pt_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger('MMSA')

        while True:
            epochs += 1
            y_pred, y_true = [], []

            for mod in model:
                mod.train()

            train_loss = 0.0
            left = self.args.update_epochs

            with tqdm(dataloader['train'], desc=f"Ep {epochs}") as td:
                for bd in td:
                    if left == self.args.update_epochs:
                        opt.zero_grad()
                        left -= 1

                    T = bd['text'].to(self.device)
                    A = bd['audio'].to(self.device)
                    V = bd['vision'].to(self.device)
                    L = bd['labels']['M'].to(self.device).view(-1, 1)

                    out = mdl(T, A, V, is_distill=True)

                    lh = torch.stack([out['logits_l_homo'], out['logits_v_homo'], out['logits_a_homo']])
                    rh = torch.stack([out['repr_l_homo'], out['repr_v_homo'], out['repr_a_homo']])
                    lhe = torch.stack([out['logits_l_hetero'], out['logits_v_hetero'], out['logits_a_hetero']])
                    rhe = torch.stack([out['repr_l_hetero'], out['repr_v_hetero'], out['repr_a_hetero']])

                    eg_h, _ = homo(lh, rh)
                    eg_he, _ = hete(lhe, rhe)

                    # Task loss (8 terms)
                    lt = self.criterion(out['output_logit'], L)
                    lt_l = self.criterion(out['logits_l_homo'], L)
                    lt_v = self.criterion(out['logits_v_homo'], L)
                    lt_a = self.criterion(out['logits_a_homo'], L)
                    lt_lh = self.criterion(out['logits_l_hetero'], L)
                    lt_vh = self.criterion(out['logits_v_hetero'], L)
                    lt_ah = self.criterion(out['logits_a_hetero'], L)
                    lt_c = self.criterion(out['logits_c'], L)
                    loss_task = lt + lt_l + lt_v + lt_a + lt_lh + lt_vh + lt_ah + lt_c

                    # Reconstruction loss
                    lr = self.MSE(out['recon_l'], out['origin_l'])
                    lr = lr + self.MSE(out['recon_v'], out['origin_v'])
                    lr = lr + self.MSE(out['recon_a'], out['origin_a'])

                    # Cycle consistency
                    ls = self.MSE(out['s_l'].permute(1, 2, 0), out['s_l_r'])
                    ls = ls + self.MSE(out['s_v'].permute(1, 2, 0), out['s_v_r'])
                    ls = ls + self.MSE(out['s_a'].permute(1, 2, 0), out['s_a_r'])

                    # Orthogonal loss
                    B = L.size(0)
                    neg = torch.full((B,), -1.0, device=self.device)
                    sl = out['s_l'].transpose(0, 1).reshape(B, -1)
                    cl = out['c_l'].transpose(0, 1).reshape(B, -1)
                    sv = out['s_v'].transpose(0, 1).reshape(B, -1)
                    cv = out['c_v'].transpose(0, 1).reshape(B, -1)
                    sa = out['s_a'].transpose(0, 1).reshape(B, -1)
                    ca = out['c_a'].transpose(0, 1).reshape(B, -1)
                    lo = self.cosine(sl, cl, neg)
                    # FIX: use non-inplace addition for autograd compatibility
                    lo = lo + self.cosine(sv, cv, neg)
                    lo = lo + self.cosine(sa, ca, neg)

                    # Margin loss
                    fl, il = [], []
                    for i in range(B):
                        fl.extend([out['c_l_sim'][i:i+1], out['c_v_sim'][i:i+1], out['c_a_sim'][i:i+1]])
                        il.extend([L[i:i+1]] * 3)
                    feats = torch.cat(fl, dim=0)
                    ids = torch.cat(il, dim=0)
                    lsim = self.sim_loss(ids, feats)

                    # GD losses
                    reg_h, ll_h, rr_h = homo.distillation_loss(lh, rh, eg_h)
                    graph_homo = 0.05 * (ll_h + reg_h)
                    reg_he, ll_he, rr_he = hete.distillation_loss(lhe, rhe, eg_he)
                    graph_hete = 0.05 * (ll_he + rr_he + reg_he)

                    # Total loss
                    total = loss_task + graph_homo + graph_hete + (ls + lr + (lsim + lo) * 0.1) * 0.1

                    # DEBUG: Loss breakdown (first batch of first epoch)
                    if epochs == 1 and left == self.args.update_epochs:
                        print(f"\n[DEBUG] === Loss Breakdown (Ep1, Batch1) ===")
                        print(f"[DEBUG] loss_task     = {loss_task.item():.4f}")
                        print(f"[DEBUG]   lt(output)  = {lt.item():.4f}")
                        print(f"[DEBUG]   lt_l_homo   = {lt_l.item():.4f}")
                        print(f"[DEBUG]   lt_v_homo   = {lt_v.item():.4f}")
                        print(f"[DEBUG]   lt_a_homo   = {lt_a.item():.4f}")
                        print(f"[DEBUG]   lt_l_hetero = {lt_lh.item():.4f}")
                        print(f"[DEBUG]   lt_v_hetero = {lt_vh.item():.4f}")
                        print(f"[DEBUG]   lt_a_hetero = {lt_ah.item():.4f}")
                        print(f"[DEBUG]   lt_c        = {lt_c.item():.4f}")
                        print(f"[DEBUG] graph_homo    = {graph_homo:.4f}")
                        print(f"[DEBUG] graph_hete    = {graph_hete:.4f}")
                        print(f"[DEBUG] ls (cycle)    = {ls:.4f}")
                        print(f"[DEBUG] lr (recon)    = {lr:.4f}")
                        print(f"[DEBUG] lsim (margin) = {lsim.item():.4f}")
                        print(f"[DEBUG] lo (orthog)   = {lo.item():.4f}")
                        print(f"[DEBUG] TOTAL         = {total.item():.4f}")
                        print(f"[DEBUG] ====================================\n")

                    total.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += total.item()
                    y_pred.append(out['output_logit'].cpu())
                    y_true.append(L.cpu())

                    if not left:
                        opt.step()
                        left = self.args.update_epochs

            if not left:
                opt.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            tr = self.metrics(pred, true)
            tr["Loss"] = round(train_loss, 4)
            logger.info(f"TRAIN >> {self._str(tr)}")

            vr = self.do_test(mdl, dataloader['valid'], mode="VAL")
            ter = self.do_test(mdl, dataloader['test'], mode="TEST")
            cur_v = vr[self.args.KeyEval]
            sched.step(vr['Loss'])

            torch.save(mdl.state_dict(), pt_dir / f"{epochs}.pth")

            better = (cur_v <= best_valid - 1e-6) if minmax == 'min' else (cur_v >= best_valid + 1e-6)
            if better:
                best_valid, best_epoch = cur_v, epochs
                torch.save(mdl.state_dict(), pt_dir / self.args.save_name)

            logger.info(f"Ep {epochs}/{epochs-best_epoch} | Best:{best_epoch} | "
                        f"VAL Loss={vr['Loss']:.4f} Acc7={vr['Acc_7']:.2%} | "
                        f"TEST Acc7={ter['Acc_7']:.2%}")

            if return_epoch_results:
                tr["Loss"] = train_loss
                epoch_results['train'].append(tr)
                epoch_results['valid'].append(vr)
                epoch_results['test'].append(ter)

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader, desc=mode) as td:
                for bd in td:
                    T = bd['text'].to(self.device)
                    A = bd['audio'].to(self.device)
                    V = bd['vision'].to(self.device)
                    L = bd['labels']['M'].to(self.device).view(-1, 1)

                    out = model(T, A, V, is_distill=True)
                    eval_loss += self.criterion(out['output_logit'], L).item()
                    y_pred.append(out['output_logit'].cpu())
                    y_true.append(L.cpu())

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            er = self.metrics(pred, true)
            er["Loss"] = round(eval_loss / len(dataloader), 4)
            logger = logging.getLogger('MMSA')
            logger.info(f"{mode} >> {self._str(er)}")

        return er


# =============================================================================
# PART 7: DATA LOADER
# =============================================================================

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        with open(args.featurePath, 'rb') as f:
            d = pickle.load(f, encoding='latin1')

        self.text = d[mode]['text'].astype(np.float32)
        self.vision = d[mode]['vision'].astype(np.float32)
        self.audio = d[mode]['audio'].astype(np.float32)
        self.labels = {'M': np.array(d[mode]['regression_labels']).astype(np.float32)}

        self.audio[self.audio == -np.inf] = 0

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, idx):
        return {
            'text': torch.Tensor(self.text[idx]),
            'audio': torch.Tensor(self.audio[idx]),
            'vision': torch.Tensor(self.vision[idx]),
            'labels': {k: torch.Tensor(v[idx].reshape(-1)) for k, v in self.labels.items()}
        }


def MMDataLoader(args, num_workers=0):
    ds = {m: MMDataset(args, m) for m in ['train', 'valid', 'test']}
    return {m: DataLoader(ds[m], batch_size=args.batch_size, num_workers=num_workers, shuffle=(m == 'train'))
            for m in ds}


# =============================================================================
# PART 8: MAIN
# =============================================================================

def get_args(mode='aligned', seed=42, epochs=30, batch=16, lr=0.0001):
    aligned = (mode == 'aligned')
    return Namespace(
        dataset_name='mosi',
        model_name='dmd',
        featurePath=f"./data/{mode}_50.pkl",
        need_data_aligned=aligned,
        batch_size=batch,
        learning_rate=lr,
        early_stop=15,
        use_bert=False,
        use_finetune=False,
        text_dropout=0.5,
        attn_dropout=0.3,
        attn_dropout_a=0.2,
        attn_dropout_v=0.0,
        output_dropout=0.5,
        relu_dropout=0.0,
        embed_dropout=0.2,
        res_dropout=0.0,
        dst_feature_dim_nheads=[50, 10],
        nlevels=4,
        conv1d_kernel_size_l=5,
        conv1d_kernel_size_a=5,
        conv1d_kernel_size_v=5,
        grad_clip=0.6,
        patience=5,
        weight_decay=0.0,
        feature_dims=[768, 5, 20],
        train_mode='regression',
        KeyEval='Loss',
        update_epochs=1,
        attn_mask=False,
        pretrained='bert-base-uncased',
        cur_seed=seed,
    )


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_one_mode(mode, seeds, epochs, batch, lr, lg, results_dir=None):
    mode_name = mode.upper()

    # FIX: log path corrected to ./logs/dmd/
    ld = Path("./logs/dmd")
    ld.mkdir(parents=True, exist_ok=True)
    for h in lg.handlers[:]:
        lg.removeHandler(h)
    fh = logging.FileHandler(ld / f"dmd_{mode_name}_{'_'.join(map(str,seeds))}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    lg.addHandler(fh)
    lg.addHandler(ch)

    lg.info("=" * 70)
    lg.info(f"DMD BASELINE -- {mode_name} | Seeds: {seeds} | Epochs: {epochs} | "
            f"Batch: {batch} | LR: {lr}")
    lg.info("DMD architecture: exact match to official trains/singleTask/model/dmd.py")
    lg.info("=" * 70)

    all_results = []
    for seed in seeds:
        lg.info("-" * 70)
        lg.info(f">>> Seed {seed} <<<")
        setup_seed(seed)
        args = get_args(mode, seed, epochs, batch, lr)
        args.save_name = f"dmd_{mode}_{seed}.pth"

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        lg.info(f"Device: {dev}")

        dl = MMDataLoader(args, num_workers=0)
        DST = 50
        mdl = DMDModel(args).to(dev)
        homo = DistillationKernelHomo(
            n_classes=1, hidden_size=DST, gd_size=64,
            to_idx=[0, 1, 2], from_idx=[0, 1, 2],
            gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
            gd_reg=10, w_losses=[1, 10], metric='l1', alpha=1/8).to(dev)
        hete = DistillationKernelHetero(
            n_classes=1, hidden_size=DST * 2, gd_size=32,
            to_idx=[0, 1, 2], from_idx=[0, 1, 2],
            gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
            gd_reg=10, w_losses=[1, 10], metric='l1', alpha=1/8).to(dev)

        total = sum(p.numel() for m in [mdl, homo, hete] for p in m.parameters())
        lg.info(f"Parameters: {total:,}")

        trainer = DMDTrainer(args, dev)
        trainer.do_train([mdl, homo, hete], dl)

        # FIX: weight load path corrected to ./pt/dmd/
        mdl.load_state_dict(torch.load(f"./pt/dmd/{args.save_name}"))
        r = trainer.do_test(mdl, dl['test'], mode="FINAL")
        lg.info(f"Seed {seed} RESULT: Acc7={r.get('Acc_7',0):.2%} Acc2={r.get('Acc_2',0):.2%} "
                f"F1={r.get('F1_score',0):.2%} MAE={r.get('MAE',0):.4f} Corr={r.get('Corr',0):.4f}")
        all_results.append({'seed': seed, **r})

        # Save metrics.json for visualization compatibility
        if results_dir is not None:
            exp_dir = Path(results_dir) / f"dmd_{mode}_seed{seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            metrics_data = {
                'model': 'dmd',
                'mode': mode,
                'seed': seed,
                'acc7': r.get('Acc_7', 0),
                'acc2': r.get('Acc_2', 0),
                'f1': r.get('F1_score', 0),
                'mae': r.get('MAE', 0),
                'corr': r.get('Corr', 0),
            }
            json_path = exp_dir / "metrics.json"
            with open(json_path, 'w') as jf:
                json.dump(metrics_data, jf, indent=2)
            lg.info(f"Saved metrics to: {json_path}")

        del mdl, homo, hete, trainer, dl
        torch.cuda.empty_cache()
        gc.collect()

    return all_results


def print_summary(lg, all_results, mode_label, results_dir=None):
    REPORT_KEYS = ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']
    seeds = [r['seed'] for r in all_results]
    n_seeds = len(all_results)

    lg.info("=" * 70)
    lg.info(f"INDIVIDUAL SEEDS -- {mode_label}")
    lg.info("-" * 70)
    header = f"{'Seed':<8}"
    for k in REPORT_KEYS:
        header += f" | {k:<10}"
    lg.info(header)
    lg.info("-" * 70)
    for r in all_results:
        row = f"{r['seed']:<8}"
        for k in REPORT_KEYS:
            row += f" | {r.get(k,0):<10.4f}"
        lg.info(row)

    lg.info("=" * 70)
    lg.info(f"AGGREGATE STATISTICS -- {mode_label}")
    lg.info("-" * 70)
    for key in REPORT_KEYS:
        vals = [r[key] for r in all_results]
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1) if n_seeds > 1 else 0.0
        raw = ", ".join(f"{v:.4f}" for v in vals)
        lg.info(f"{key}: Mean={mu:.4f} | Std={sd:.4f} | Raw=[{raw}]")

    if n_seeds >= 2:
        lg.info("=" * 70)
        lg.info(f"STATISTICAL TESTS -- {mode_label}")
        lg.info("-" * 70)
        for key in REPORT_KEYS:
            vals = np.array([r[key] for r in all_results])
            mu = np.mean(vals)
            sd = np.std(vals, ddof=1)
            cv = abs(sd / mu) if mu != 0 else 0.0
            stability = "stable" if cv < 0.05 else "moderate" if cv < 0.10 else "unstable"
            lg.info(f"  {key}: CV={cv:.4f} ({stability})")

        lg.info("-" * 70)
        lg.info(" One-sample t-test vs DMD Paper Baseline (H0: mean = baseline)")
        ref_acc7 = 0.414 if 'ALIGNED' in mode_label.upper() else 0.408
        ref_acc2 = 0.847 if 'ALIGNED' in mode_label.upper() else 0.839
        for key, ref in [('Acc_7', ref_acc7), ('Acc_2', ref_acc2)]:
            vals = np.array([r[key] for r in all_results])
            t_stat, p_val = stats.ttest_1samp(vals, ref)
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            lg.info(f"  {key}: t={t_stat:.4f}, p={p_val:.4f} {significance} "
                    f"(baseline={ref:.3f}, observed={np.mean(vals):.4f})")

    # Save to results txt file
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        fname = results_dir / f"results_{mode_label.lower()}.txt"
        with open(fname, 'w') as f:
            f.write(f"DMD BASELINE -- {mode_label}\n")
            f.write("=" * 70 + "\n")
            f.write(f"INDIVIDUAL SEEDS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Seed':<8}" + "".join(f" | {k:<10}" for k in REPORT_KEYS) + "\n")
            for r in all_results:
                row = f"{r['seed']:<8}"
                for k in REPORT_KEYS:
                    row += f" | {r.get(k,0):<10.4f}"
                f.write(row + "\n")
            f.write("=" * 70 + "\n")
            f.write(f"AGGREGATE STATISTICS\n")
            f.write("-" * 70 + "\n")
            for key in REPORT_KEYS:
                vals = [r[key] for r in all_results]
                mu = np.mean(vals)
                sd = np.std(vals, ddof=1) if n_seeds > 1 else 0.0
                raw = ", ".join(f"{v:.4f}" for v in vals)
                f.write(f"{key}: Mean={mu:.4f} | Std={sd:.4f} | Raw=[{raw}]\n")
            if n_seeds >= 2:
                f.write("=" * 70 + "\n")
                f.write("STATISTICAL TESTS\n")
                f.write("-" * 70 + "\n")
                for key in REPORT_KEYS:
                    vals = np.array([r[key] for r in all_results])
                    mu = np.mean(vals)
                    sd = np.std(vals, ddof=1)
                    cv = abs(sd / mu) if mu != 0 else 0.0
                    stability = "stable" if cv < 0.05 else "moderate" if cv < 0.10 else "unstable"
                    f.write(f"  {key}: CV={cv:.4f} ({stability})\n")
                f.write("-" * 70 + "\n")
                ref_acc7 = 0.414 if 'ALIGNED' in mode_label.upper() else 0.408
                ref_acc2 = 0.847 if 'ALIGNED' in mode_label.upper() else 0.839
                for key, ref in [('Acc_7', ref_acc7), ('Acc_2', ref_acc2)]:
                    vals = np.array([r[key] for r in all_results])
                    t_stat, p_val = stats.ttest_1samp(vals, ref)
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                    f.write(f"  {key}: t={t_stat:.4f}, p={p_val:.4f} {significance} "
                            f"(baseline={ref:.3f}, observed={np.mean(vals):.4f})\n")


def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument('mode', nargs='?', default='aligned',
                    choices=['aligned', 'unaligned', 'all'],
                    help='aligned | unaligned | all (both sequentially, aligned first)')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--epochs', type=int, default=30)
    pa.add_argument('--batch_size', type=int, default=16)
    pa.add_argument('--lr', type=float, default=0.0001)
    pa.add_argument('--multi_seed', action='store_true',
                    help='Run all 4 seeds: 42, 1111, 1112, 1113')

    a = pa.parse_args(sys.argv[1:])

    SEEDS = [42, 1111, 1112, 1113] if a.multi_seed else [a.seed]
    epochs = a.epochs
    batch = a.batch_size
    lr = a.lr

    lg = logging.getLogger('MMSA')
    lg.setLevel(logging.DEBUG)
    for h in lg.handlers[:]:
        lg.removeHandler(h)

    # FIX: results path corrected to ./results/dmd/
    results_dir = Path("./results/dmd")
    results_dir.mkdir(parents=True, exist_ok=True)

    if a.mode == 'all':
        lg.info("")
        lg.info("#" * 70)
        lg.info("# STEP 1/2: ALIGNED MODE (4 seeds)")
        lg.info("#" * 70)

        aligned_results = run_one_mode('aligned', SEEDS, epochs, batch, lr, lg, results_dir)
        print_summary(lg, aligned_results, "ALIGNED", results_dir)

        lg.info("=" * 70)
        lg.info("DMD PAPER BASELINE COMPARISON -- ALIGNED")
        lg.info("-" * 70)
        lg.info(f"{'Metric':<12} | {'Ours (DMD)':<18} | {'Paper (DMD)':<15} | {'Gap':<8}")
        lg.info("-" * 70)
        for key in ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']:
            ours = np.mean([r[key] for r in aligned_results])
            ref_map = {'Acc_7': 0.414, 'Acc_2': 0.847, 'F1_score': 0.843, 'MAE': 1.156, 'Corr': 0.704}
            ref = ref_map.get(key, 0.0)
            gap = ours - ref
            sig = "*" if abs(gap) > 0.02 else ""
            lg.info(f"{key:<12} | {ours:.4f} +/- {np.std([r[key] for r in aligned_results], ddof=1):.4f} | "
                    f"{ref:.4f} | {gap:+.4f}{sig}")

        lg.info("")
        lg.info("#" * 70)
        lg.info("# STEP 2/2: UNALIGNED MODE (4 seeds)")
        lg.info("#" * 70)

        unaligned_results = run_one_mode('unaligned', SEEDS, epochs, batch, lr, lg, results_dir)
        print_summary(lg, unaligned_results, "UNALIGNED", results_dir)

        lg.info("")
        lg.info("#" * 70)
        lg.info("# FINAL SUMMARY: ALIGNED vs UNALIGNED")
        lg.info("#" * 70)
        lg.info("=" * 70)
        lg.info(f"{'Metric':<12} | {'Aligned Mean+/-Std':<22} | {'Unaligned Mean+/-Std':<24} | {'Diff':<8} | {'p-value':<10}")
        lg.info("-" * 70)
        for key in ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']:
            a_vals = np.array([r[key] for r in aligned_results])
            u_vals = np.array([r[key] for r in unaligned_results])
            a_mu = np.mean(a_vals)
            a_sd = np.std(a_vals, ddof=1)
            u_mu = np.mean(u_vals)
            u_sd = np.std(u_vals, ddof=1)
            diff = u_mu - a_mu
            if len(a_vals) >= 2 and len(u_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(a_vals, u_vals, equal_var=False)
                sig = "*" if p_val < 0.05 else "**" if p_val < 0.01 else "***" if p_val < 0.001 else "n.s."
            else:
                p_val = float('nan')
                sig = "N/A"
            lg.info(f"{key:<12} | {a_mu:.4f}+/-{a_sd:.4f} | {u_mu:.4f}+/-{u_sd:.4f} | "
                    f"{diff:+.4f} | {p_val:.4f} {sig}")
        lg.info("=" * 70)

        lg.info("DMD PAPER BASELINE COMPARISON -- UNALIGNED")
        lg.info("-" * 70)
        lg.info(f"{'Metric':<12} | {'Ours (DMD)':<18} | {'Paper (DMD)':<15} | {'Gap':<8}")
        lg.info("-" * 70)
        for key in ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']:
            ours = np.mean([r[key] for r in unaligned_results])
            ref_map = {'Acc_7': 0.408, 'Acc_2': 0.839, 'F1_score': 0.835, 'MAE': 1.177, 'Corr': 0.695}
            ref = ref_map.get(key, 0.0)
            gap = ours - ref
            sig = "*" if abs(gap) > 0.02 else ""
            lg.info(f"{key:<12} | {ours:.4f} +/- {np.std([r[key] for r in unaligned_results], ddof=1):.4f} | "
                    f"{ref:.4f} | {gap:+.4f}{sig}")

        # Save final summary to txt
        final_fname = results_dir / "final_summary.txt"
        with open(final_fname, 'w') as f:
            f.write("DMD BASELINE FINAL SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("DMD PAPER BASELINE COMPARISON -- ALIGNED\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Metric':<12} | {'Ours (DMD)':<18} | {'Paper (DMD)':<15} | {'Gap':<8}\n")
            for key in ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']:
                ours = np.mean([r[key] for r in aligned_results])
                ref_map = {'Acc_7': 0.414, 'Acc_2': 0.847, 'F1_score': 0.843, 'MAE': 1.156, 'Corr': 0.704}
                ref = ref_map.get(key, 0.0)
                gap = ours - ref
                sig = "*" if abs(gap) > 0.02 else ""
                f.write(f"{key:<12} | {ours:.4f} +/- {np.std([r[key] for r in aligned_results], ddof=1):.4f} | "
                        f"{ref:.4f} | {gap:+.4f}{sig}\n")

            f.write("\n")
            f.write("DMD PAPER BASELINE COMPARISON -- UNALIGNED\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Metric':<12} | {'Ours (DMD)':<18} | {'Paper (DMD)':<15} | {'Gap':<8}\n")
            for key in ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']:
                ours = np.mean([r[key] for r in unaligned_results])
                ref_map = {'Acc_7': 0.408, 'Acc_2': 0.839, 'F1_score': 0.835, 'MAE': 1.177, 'Corr': 0.695}
                ref = ref_map.get(key, 0.0)
                gap = ours - ref
                sig = "*" if abs(gap) > 0.02 else ""
                f.write(f"{key:<12} | {ours:.4f} +/- {np.std([r[key] for r in unaligned_results], ddof=1):.4f} | "
                        f"{ref:.4f} | {gap:+.4f}{sig}\n")

            f.write("\n")
            f.write("FINAL SUMMARY: ALIGNED vs UNALIGNED\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Metric':<12} | {'Aligned Mean+/-Std':<22} | {'Unaligned Mean+/-Std':<24} | {'Diff':<8} | {'p-value':<10}\n")
            f.write("-" * 70 + "\n")
            for key in ['Acc_7', 'Acc_2', 'F1_score', 'MAE', 'Corr']:
                a_vals = np.array([r[key] for r in aligned_results])
                u_vals = np.array([r[key] for r in unaligned_results])
                a_mu = np.mean(a_vals)
                a_sd = np.std(a_vals, ddof=1)
                u_mu = np.mean(u_vals)
                u_sd = np.std(u_vals, ddof=1)
                diff = u_mu - a_mu
                if len(a_vals) >= 2 and len(u_vals) >= 2:
                    t_stat, p_val = stats.ttest_ind(a_vals, u_vals, equal_var=False)
                    sig = "*" if p_val < 0.05 else "**" if p_val < 0.01 else "***" if p_val < 0.001 else "n.s."
                else:
                    p_val = float('nan')
                    sig = "N/A"
                f.write(f"{key:<12} | {a_mu:.4f}+/-{a_sd:.4f} | {u_mu:.4f}+/-{u_sd:.4f} | "
                        f"{diff:+.4f} | {p_val:.4f} {sig}\n")

        lg.info(f"Results saved to ./results/dmd/results_aligned.txt, ./results/dmd/results_unaligned.txt, ./results/dmd/final_summary.txt")

    else:
        results = run_one_mode(a.mode, SEEDS, epochs, batch, lr, lg, results_dir)
        print_summary(lg, results, a.mode.upper(), results_dir)


if __name__ == "__main__":
    main()
