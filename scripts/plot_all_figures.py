#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_all_figures.py - MER-MTL Paper Figure Generator
======================================================
Generates two publication-quality figures:

Figure 1 - Training Curves (2x2 layout):
    (0,0) Aligned + TT      (0,1) Aligned + MP
    (1,0) Unaligned + TT    (1,1) Unaligned + MP

Figure 2 - Performance Comparison (3+2 layout):
    Top row:    Acc-7   | Acc-2   | F1
    Bottom row: MAE (centered) | Corr (centered)

Usage:
    python plot_all_figures.py --results_dir ./results --logs_dir ./logs/mermtl --output_dir ./figures
"""
import os
import sys
import json
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.visualization import _collect_log_paths, _parse_log_sigmas, AUX_TASK_NAMES, AUX_COLORS

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = ['seed42', 'seed1111', 'seed1112', 'seed1113']
MODES = ['aligned', 'unaligned']
METRICS = ['acc7', 'acc2', 'f1', 'mae', 'corr']
METRIC_LABELS = {
    'acc7': 'Acc-7 (%)',
    'acc2': 'Acc-2 (%)',
    'f1':   'F1 (%)',
    'mae':  'MAE',
    'corr': 'Corr',
}

COLORS = {
    'DMD':        '#34495E',
    'MER_MTL_TT': '#2980B9',
    'MER_MTL_MP': '#E67E22',
}

DISPLAY_NAMES = {
    'DMD':        'DMD (reimpl.)',
    'MER_MTL_TT': 'MER-MTL-TT',
    'MER_MTL_MP': 'MER-MTL-MP',
}

# AUX_WEIGHT removed: weight calculation now uses normalized formula (inv_sigma_sq / sum(inv_sigma_sq))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _empty_mode_dict():
    return {mode: {m: {} for m in METRICS} for mode in MODES}


def _load_metrics_json(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    return data


def _infer_model_key(data, dir_path):
    model_name = str(data.get('model', '')).lower()
    text_mode = str(data.get('text_mode', '')).lower()
    if 'mer' in model_name or 'mtl' in model_name:
        if 'mp' in text_mode:
            return 'MER_MTL_MP'
        return 'MER_MTL_TT'
    if 'dmd' in model_name:
        return 'DMD'
    path_lower = dir_path.lower()
    if 'mer_mtl_tt' in path_lower or 'mer-mtl-tt' in path_lower:
        return 'MER_MTL_TT'
    if 'mer_mtl_mp' in path_lower or 'mer-mtl-mp' in path_lower:
        return 'MER_MTL_MP'
    if 'mer_mtl' in path_lower or 'mer-mtl' in path_lower:
        if '_mp_' in path_lower or '/mp/' in path_lower:
            return 'MER_MTL_MP'
        return 'MER_MTL_TT'
    if 'dmd' in path_lower:
        return 'DMD'
    return None


def _infer_mode_from_path(dir_path):
    path_lower = dir_path.lower()
    if 'unaligned' in path_lower:
        return 'unaligned'
    if 'aligned' in path_lower:
        return 'aligned'
    return None


def _infer_seed_from_path(dir_path):
    m = re.search(r'seed(\d+)', dir_path, re.IGNORECASE)
    if m:
        return f"seed{m.group(1)}"
    return ''


def load_mermtl_from_dir(base_dir):
    results = {}
    mermtl_dir = os.path.join(base_dir, 'mermtl')
    search_dir = mermtl_dir if os.path.isdir(mermtl_dir) else base_dir

    for root, dirs, files in os.walk(search_dir):
        if 'metrics.json' not in files:
            continue
        metrics_path = os.path.join(root, 'metrics.json')
        data = _load_metrics_json(metrics_path)
        if data is None:
            continue
        model_key = _infer_model_key(data, root)
        if model_key not in ('MER_MTL_TT', 'MER_MTL_MP'):
            continue
        mode = data.get('mode')
        if mode not in MODES:
            mode = _infer_mode_from_path(root)
        if mode not in MODES:
            continue
        seed_val = data.get('seed')
        seed_key = f"seed{seed_val}" if seed_val is not None else _infer_seed_from_path(root)
        if seed_key not in SEEDS:
            continue
        if model_key not in results:
            results[model_key] = _empty_mode_dict()
        for metric in METRICS:
            val = data.get(metric)
            if val is not None:
                if metric in ('acc7', 'acc2', 'f1'):
                    results[model_key][mode][metric][seed_key] = float(val) * 100
                else:
                    results[model_key][mode][metric][seed_key] = float(val)
    return results


def parse_dmd_results_txt(txt_path):
    if not os.path.exists(txt_path):
        return {}
    with open(txt_path, 'r') as f:
        content = f.read()
    seed_data = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('Seed') or line.startswith('-'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6:
            continue
        try:
            seed_num = int(parts[0])
        except ValueError:
            continue
        seed_key = f"seed{seed_num}"
        if seed_key not in SEEDS:
            continue
        try:
            acc7 = float(parts[1])
            acc2 = float(parts[2])
            f1   = float(parts[3])
            mae  = float(parts[4])
            corr = float(parts[5])
        except ValueError:
            continue
        if acc7 <= 1.0:
            acc7 *= 100.0
        if acc2 <= 1.0:
            acc2 *= 100.0
        if f1   <= 1.0:
            f1   *= 100.0
        seed_data[seed_key] = {'acc7': acc7, 'acc2': acc2, 'f1': f1, 'mae': mae, 'corr': corr}
    return seed_data


def load_dmd_from_dir(base_dir):
    dmd_dir = os.path.join(base_dir, 'dmd')
    result = {'aligned': {}, 'unaligned': {}}
    for mode in MODES:
        candidates = [
            os.path.join(dmd_dir, f'results_{mode}.txt'),
            os.path.join(base_dir, f'results_{mode}.txt'),
        ]
        for fpath in candidates:
            if os.path.exists(fpath):
                result[mode] = parse_dmd_results_txt(fpath)
                break
    return result


def build_results(results_dir):
    results = {}
    mermtl_data = load_mermtl_from_dir(results_dir)
    for model_key in mermtl_data:
        results[model_key] = mermtl_data[model_key]
    dmd_data = load_dmd_from_dir(results_dir)
    if dmd_data['aligned'] or dmd_data['unaligned']:
        results['DMD'] = _empty_mode_dict()
        for mode in MODES:
            for seed_key, seed_vals in dmd_data[mode].items():
                for metric in METRICS:
                    if metric in seed_vals:
                        results['DMD'][mode][metric][seed_key] = seed_vals[metric]
    return results


def _mean_std(d):
    if not d:
        return None, None
    vals = list(d.values())
    return np.mean(vals), np.std(vals)


# ---------------------------------------------------------------------------
# Figure 1: Training Curves (2x2)
# ---------------------------------------------------------------------------
def _compute_epoch_weights(log_files_by_mode, mode, variant):
    """Compute per-epoch effective loss weights for a given mode/variant."""
    mode_logs = log_files_by_mode.get(mode, {})
    logs = mode_logs.get(variant, [])
    all_sigma_data = []
    for log_path in logs:
        data = _parse_log_sigmas(log_path)
        if data:
            all_sigma_data.append(data)
    if not all_sigma_data:
        return None, None, None

    num_tasks = len(all_sigma_data[0][0]['sigmas'])
    MAX_EPOCH = 30  # Training runs 30 epochs; truncate visualization
    epoch_weights = {ep: [[] for _ in range(num_tasks)]
                     for ep in range(1, MAX_EPOCH + 1)}

    for sd in all_sigma_data:
        for item in sd:
            ep = item['epoch']
            if ep > MAX_EPOCH:
                continue
            sigmas = item['sigmas']
            inv_sigma_sq = [1.0 / (s ** 2) for s in sigmas]
            total_inv = sum(inv_sigma_sq)
            for k in range(min(num_tasks, len(sigmas))):
                eff_weight = inv_sigma_sq[k] / total_inv
                epoch_weights[ep][k].append(eff_weight)

    return epoch_weights, MAX_EPOCH, num_tasks


def plot_training_curves_figure(log_files_by_mode, output_dir):
    """Generate 2x2 training curves figure (screenshot style).

    Layout:
        (0,0) Aligned + TT      (0,1) Aligned + MP
        (1,0) Unaligned + TT    (1,1) Unaligned + MP

    Each subplot has its OWN legend showing 4 task colors.
    COLOR = task identity (L_rec/L_cyc/L_mar/L_ort).
    No line style differentiation; all solid lines.
    """
    variants = [
        ('MER-MTL-TT', 'MER-MTL-TT'),
        ('MER-MTL-MP', 'MER-MTL-MP'),
    ]
    modes_grid = ['aligned', 'unaligned']

    # Task colors (matching screenshot: red, blue, green, orange)
    TASK_COLORS = {
        'L_rec': '#E74C3C',
        'L_cyc': '#3498DB',
        'L_mar': '#2ECC71',
        'L_ort': '#F39C12',
    }

    # Check data availability
    has_data = False
    for mode in modes_grid:
        for _, var_key in variants:
            if log_files_by_mode.get(mode, {}).get(var_key):
                has_data = True
                break
    if not has_data:
        print("  WARNING: No training log data available for curves figure.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, mode in enumerate(modes_grid):
        for col, (var_label, var_key) in enumerate(variants):
            ax = axes[row][col]
            result = _compute_epoch_weights(log_files_by_mode, mode, var_key)
            epoch_weights, max_epoch, num_tasks = result

            if epoch_weights is None:
                ax.set_visible(False)
                continue

            task_names = AUX_TASK_NAMES[:num_tasks]
            x = list(range(1, max_epoch + 1))

            for k, name in enumerate(task_names):
                color = TASK_COLORS.get(name, '#333333')
                means = [np.mean(epoch_weights[ep][k])
                         if epoch_weights[ep][k] else 0 for ep in x]
                stds = [np.std(epoch_weights[ep][k])
                        if epoch_weights[ep][k] else 0 for ep in x]

                ax.plot(x, means, '-', color=color, linewidth=2.0,
                        label=name, zorder=3)
                ax.fill_between(x,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.12, color=color, zorder=2)

            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Eff. Loss Weight', fontsize=9, labelpad=6)
            ax.set_title(f'{mode.capitalize()} / {var_label}',
                         fontsize=10, fontweight='bold')
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            # Independent legend per subplot (upper-left)
            ax.legend(loc='upper left', fontsize=7.5, frameon=True,
                      edgecolor='gray', fancybox=False, facecolor='white',
                      framealpha=0.85)

    # Single-border frames around each subplot (just axes spines, no double border)
    for ax in axes.flatten():
        if not ax.get_visible():
            continue
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path_png = os.path.join(output_dir, 'fig_training_curves.png')
    path_pdf = os.path.join(output_dir, 'fig_training_curves.pdf')
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path_png}")
    print(f"  Saved: {path_pdf}")
    return path_pdf


# ---------------------------------------------------------------------------
# Figure 2: Performance Comparison (3+2 layout)
# ---------------------------------------------------------------------------
def plot_performance_figure(results, output_dir):
    """Generate performance comparison figure with 3+2 layout.

    Top row (3 subplots):    Acc-7  |  Acc-2  |  F1
    Bottom row (2 subplots): MAE  |  Corr  (centered, equal gap)

    Each subplot shows grouped bars for 3 models x 2 modes.
    COLOR distinguishes models: DMD / MER-MTL-TT / MER-MTL-MP.
    STYLE distinguishes modes: Aligned = solid, Unaligned = hatched.
    All 5 subplots identical in size; inter-subplot gap equal across
    both rows; bottom row centered.

    Uses fig.add_axes() for precise positioning (GridSpec cannot
    produce equal gaps with different subplot counts per row).
    """
    models = [m for m in ['DMD', 'MER_MTL_TT', 'MER_MTL_MP'] if m in results]
    if len(models) < 2:
        print("  WARNING: Need >= 2 models for performance figure.")
        return None

    fig = plt.figure(figsize=(10, 7.5))

    # ---- Manual axes positioning for equal gaps & centered bottom row ----
    # Key insight: if both rows span the same total width, then
    #   3w + 2g = 2w + G  =>  w = G (gap equals subplot width)
    # So the bottom gap automatically equals the top gap.
    fig_w, fig_h = 10.0, 7.5

    left_in  = 0.80    # left margin (inches, room for y-labels)
    right_in = 0.60    # right margin
    avail_w  = fig_w - left_in - right_in    # 9.60
    sub_w    = avail_w / 5.0                 # 1.92  (subplot width)
    sub_gap  = sub_w                          # 1.92  (inter-subplot gap = width)
    step     = sub_w + sub_gap                # 3.84  (pitch)

    # Top row: 3 subplots, left edges at left_in + i*step
    top_xs_in = [left_in + i * step for i in range(3)]     # 1.20, 5.04, 8.88
    # Bottom row: 2 subplots, centered on the same span
    bot_start = fig_w / 2.0 - (2 * sub_w + sub_gap) / 2.0  # 3.12
    bot_xs_in = [bot_start + i * step for i in range(2)]    # 3.12, 6.96

    # Vertical layout  (more room for legend + row gap)
    top_y_in  = 3.50   # bottom of top row (inches)
    bot_y_in  = 0.50   # bottom of bottom row
    sub_h_in  = 2.40   # subplot height (0.60" gap between rows)

    # Convert to figure coordinates
    def _r(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    ax_acc7 = fig.add_axes(_r(top_xs_in[0], top_y_in, sub_w, sub_h_in))
    ax_acc2 = fig.add_axes(_r(top_xs_in[1], top_y_in, sub_w, sub_h_in))
    ax_f1   = fig.add_axes(_r(top_xs_in[2], top_y_in, sub_w, sub_h_in))
    ax_mae  = fig.add_axes(_r(bot_xs_in[0], bot_y_in, sub_w, sub_h_in))
    ax_corr = fig.add_axes(_r(bot_xs_in[1], bot_y_in, sub_w, sub_h_in))

    top_axes    = [ax_acc7, ax_acc2, ax_f1]
    top_metrics = ['acc7', 'acc2', 'f1']
    bot_axes    = [ax_mae, ax_corr]
    bot_metrics = ['mae', 'corr']

    x = np.arange(len(models))
    bar_width  = 0.32
    bar_offset = 0.18

    def _draw_grouped_bars(ax, metric):
        """Aligned (solid) + Unaligned (hatched) bars, one pair per model."""
        aligned_vals, aligned_errs = [], []
        unaligned_vals, unaligned_errs = [], []

        for model in models:
            m_a, s_a = _mean_std(results.get(model, {}).get('aligned', {}).get(metric, {}))
            m_u, s_u = _mean_std(results.get(model, {}).get('unaligned', {}).get(metric, {}))
            aligned_vals.append(m_a or 0)
            aligned_errs.append(s_a or 0)
            unaligned_vals.append(m_u or 0)
            unaligned_errs.append(s_u or 0)

        model_colors = [COLORS.get(m, '#95A5A6') for m in models]

        # Aligned bars: solid fill
        bars_a = ax.bar(x - bar_offset, aligned_vals, bar_width,
                        yerr=aligned_errs, capsize=2,
                        label='Aligned',
                        color=model_colors,
                        edgecolor='white', linewidth=0.6, alpha=0.92)

        # Unaligned bars: hatched fill
        bars_u = ax.bar(x + bar_offset, unaligned_vals, bar_width,
                        yerr=unaligned_errs, capsize=2,
                        label='Unaligned',
                        color=model_colors,
                        edgecolor='#333333', linewidth=0.7, alpha=0.78)
        for bar in bars_u:
            bar.set_hatch('///')

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models],
                           fontsize=7, rotation=15, ha='right')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=8, labelpad=8)
        ax.set_title(METRIC_LABELS[metric],
                     fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=6.5)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

        all_vals = aligned_vals + unaligned_vals
        all_errs = aligned_errs + unaligned_errs
        ym = max(v + e for v, e in zip(all_vals, all_errs)) if all_vals else 1.0
        ax.set_ylim(0, ym * 1.30)

        for bars, vals, errs in [(bars_a, aligned_vals, aligned_errs),
                                  (bars_u, unaligned_vals, unaligned_errs)]:
            for bar, v, e in zip(bars, vals, errs):
                ax.text(bar.get_x() + bar.get_width() / 2., v + e + ym * 0.02,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)

        return bars_a, bars_u

    for ax, metric in zip(top_axes, top_metrics):
        _draw_grouped_bars(ax, metric)
    for ax, metric in zip(bot_axes, bot_metrics):
        _draw_grouped_bars(ax, metric)

    # Single-border frames (just axes spines, no double border)
    all_axes = top_axes + bot_axes
    for ax in all_axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.0)

    # Legend: 6 combinations (3 models x 2 modes), no overlap
    legend_elements = []
    for m in models:
        clr = COLORS.get(m, '#95A5A6')
        legend_elements.append(
            Patch(facecolor=clr, edgecolor='white',
                  label=f'{DISPLAY_NAMES.get(m, m)} Aligned'))
        legend_elements.append(
            Patch(facecolor=clr, edgecolor='#333333', hatch='///', alpha=0.78,
                  label=f'{DISPLAY_NAMES.get(m, m)} Unaligned'))

    fig.legend(handles=legend_elements, loc='lower right',
               ncol=1, fontsize=7.5, frameon=True, edgecolor='gray',
               fancybox=False, bbox_to_anchor=(0.99, 0.01),
               columnspacing=1.0, handlelength=1.2, borderaxespad=0.3)

    path_png = os.path.join(output_dir, 'fig_performance.png')
    path_pdf = os.path.join(output_dir, 'fig_performance.pdf')
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path_png}")
    print(f"  Saved: {path_pdf}")
    return path_pdf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate all MER-MTL paper figures')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing results/')
    parser.add_argument('--logs_dir', type=str, default='./logs/mermtl',
                        help='Directory containing training logs')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load bar chart data
    print("=" * 60)
    print("Loading results data...")
    results = build_results(args.results_dir)

    for model in results:
        for mode in MODES:
            for metric in METRICS:
                m, s = _mean_std(results[model][mode][metric])
                if m is not None:
                    print(f"  {model} / {mode} / {metric}: {m:.4f} +/- {s:.4f}")

    # 2. Load training log data
    print("\nCollecting training logs...")
    patterns = [
        ('_tt_',  'MER-MTL-TT'),
        ('_mp_',  'MER-MTL-MP'),
    ]
    log_files_by_mode = _collect_log_paths(args.logs_dir, patterns)

    for mode in MODES:
        for variant, paths in log_files_by_mode.get(mode, {}).items():
            print(f"  {mode}/{variant}: {len(paths)} logs")

    # 3. Generate Figure 1: Training Curves (2x2)
    print("\n[Figure 1] Training Curves (2x2)...")
    plot_training_curves_figure(log_files_by_mode, args.output_dir)

    # 4. Generate Figure 2: Performance Comparison (3+2)
    print("\n[Figure 2] Performance Comparison (3+2)...")
    plot_performance_figure(results, args.output_dir)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
