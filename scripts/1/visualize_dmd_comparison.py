#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMD (reimpl.) vs MER-MTL Comparison Visualization (v3)
=================================================
Reuses the same directory loading logic as visualize_comparison.py (which is known to work).

Supports:
  - DMD txt files: results/dmd/results_aligned.txt, results/dmd/results_unaligned.txt
  - MER-MTL metrics.json: results/mermtl/MER_MTL_{text_mode}_{mode}_seed{N}/metrics.json

Usage:
    python visualize_dmd_comparison.py --results_dir ./results --output ./figures
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

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.visualization import plot_training_curves

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = ['seed42', 'seed1111', 'seed1112', 'seed1113']
MODES = ['aligned', 'unaligned']
METRICS = ['acc7', 'acc2', 'f1', 'mae', 'corr']
METRIC_LABELS = {
    'acc7': 'Acc-7 (%)',
    'acc2': 'Acc-2 (%)',
    'f1': 'F1 Score (%)',
    'mae': 'MAE',
    'corr': 'Correlation'
}

COLORS = {
    'DMD':        '#34495E',
    'MER_MTL_TT': '#2980B9',
    'MER_MTL_MP': '#E67E22',
    'DMD_Paper':  '#95A5A6'
}

DISPLAY_NAMES = {
    'DMD': 'DMD (reimpl.)',
    'MER_MTL_TT': 'MER-MTL-TT',
    'MER_MTL_MP': 'MER-MTL-MP',
    'DMD_Paper': 'DMD (Paper)'
}

# DMD Paper baselines (from Li et al. 2023)
DMD_PAPER = {
    'aligned':   {'acc7': 41.40, 'acc2': 84.70, 'f1': 84.30, 'mae': 1.156, 'corr': 0.704},
    'unaligned': {'acc7': 40.80, 'acc2': 83.90, 'f1': 83.50, 'mae': 1.177, 'corr': 0.695}
}


def _empty_mode_dict():
    return {mode: {m: {} for m in METRICS} for mode in MODES}


# ---------------------------------------------------------------------------
# Loader: metrics.json (same logic as visualize_comparison.py)
# ---------------------------------------------------------------------------
def _load_metrics_json(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: could not read {metrics_path}: {e}")
        return None
    return data


def _infer_model_key(data, dir_path):
    """Infer model key from metrics.json content or directory path."""
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
    """
    Walk results/mermtl/ directory and load MER-MTL metrics.json files.
    Same logic as visualize_comparison.py.

    Returns: {model_key: {mode: {metric: {seed_key: value}}}}
    Values: acc7/acc2/f1 in percentage (0-100), mae/corr raw.
    """
    results = {}
    mermtl_dir = os.path.join(base_dir, 'mermtl')

    if not os.path.isdir(mermtl_dir):
        print(f"  WARNING: MER-MTL directory not found: {mermtl_dir}")
        # Try base_dir itself
        search_dir = base_dir
    else:
        search_dir = mermtl_dir
        print(f"  Searching MER-MTL metrics in: {mermtl_dir}")

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

        print(f"    Loaded: {model_key} / {mode} / {seed_key} <- {root}")

    return results


# ---------------------------------------------------------------------------
# Loader: DMD txt files (pipe-separated per-seed table)
# ---------------------------------------------------------------------------
def parse_dmd_results_txt(txt_path):
    """
    Parse DMD results_aligned.txt or results_unaligned.txt.
    Expected format:
        Seed     | Acc_7      | Acc_2      | F1_score   | MAE        | Corr
        42       | 0.3980     | 0.7973     | 0.7695     | 0.8960     | 0.6870
    Returns: {'seed42': {'acc7': 39.80, ...}, ...}
    """
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
        # Convert 0-1 -> percentage
        if acc7 <= 1.0: acc7 *= 100.0
        if acc2 <= 1.0: acc2 *= 100.0
        if f1   <= 1.0: f1   *= 100.0
        seed_data[seed_key] = {'acc7': acc7, 'acc2': acc2, 'f1': f1, 'mae': mae, 'corr': corr}

    return seed_data


def load_dmd_from_dir(base_dir):
    """Load DMD txt results from results/dmd/."""
    dmd_dir = os.path.join(base_dir, 'dmd')
    result = {'aligned': {}, 'unaligned': {}}

    for mode in MODES:
        # Try multiple candidate filenames
        candidates = [
            os.path.join(dmd_dir, f'results_{mode}.txt'),
            os.path.join(base_dir, f'results_{mode}.txt'),
        ]
        for fpath in candidates:
            if os.path.exists(fpath):
                print(f"  Found DMD {mode}: {fpath}")
                result[mode] = parse_dmd_results_txt(fpath)
                break

    return result


# ---------------------------------------------------------------------------
# Build unified results structure
# ---------------------------------------------------------------------------
def build_results(args):
    """
    Build unified dict: {model_key: {mode: {metric: {seed_key: value}}}}
    """
    results = {}

    # 1. Load MER-MTL
    print("Loading MER-MTL metrics.json...")
    mermtl_data = load_mermtl_from_dir(args.results_dir)
    for model_key in mermtl_data:
        results[model_key] = mermtl_data[model_key]

    # 2. Load DMD
    print("\nLoading DMD txt files...")
    dmd_data = load_dmd_from_dir(args.results_dir)
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
# Plotting functions
# ---------------------------------------------------------------------------
def plot_grouped_bar(results, output_dir):
    """3-model grouped bar chart for each metric."""
    models = [m for m in ['DMD', 'MER_MTL_TT', 'MER_MTL_MP'] if m in results]
    if len(models) < 2:
        print("  Skip grouped bar: need >= 2 models")
        return None

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 7))
    if len(METRICS) == 1:
        axes = [axes]

    x = np.arange(len(models))
    width = 0.35

    for ax, metric in zip(axes, METRICS):
        aligned_vals, unaligned_vals = [], []
        aligned_errs, unaligned_errs = [], []

        for model in models:
            m_a, s_a = _mean_std(results.get(model, {}).get('aligned', {}).get(metric, {}))
            m_u, s_u = _mean_std(results.get(model, {}).get('unaligned', {}).get(metric, {}))
            aligned_vals.append(m_a or 0)
            unaligned_vals.append(m_u or 0)
            aligned_errs.append(s_a or 0)
            unaligned_errs.append(s_u or 0)

        bars_a = ax.bar(x - width/2, aligned_vals, width, yerr=aligned_errs,
                        capsize=4, label='Aligned', edgecolor='white')
        bars_u = ax.bar(x + width/2, unaligned_vals, width, yerr=unaligned_errs,
                        capsize=4, label='Unaligned', edgecolor='white', hatch='//')

        for i, (ba, bu) in enumerate(zip(bars_a, bars_u)):
            c = COLORS.get(models[i], '#95A5A6')
            ba.set_facecolor(c)
            bu.set_facecolor(c)
            bu.set_alpha(0.6)
            bu.set_hatch('//')

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        all_v2 = aligned_vals + unaligned_vals
        ym2 = max(all_v2) if all_v2 else 1.0
        ax.set_ylim(0, ym2 * 1.2)

        # Proportional label offset based on data range
        all_v = aligned_vals + unaligned_vals
        ym = max(all_v) if all_v else 1.0
        lo = max(ym * 0.05, 0.02)
        for i, (ma, mu) in enumerate(zip(aligned_vals, unaligned_vals)):
            ax.text(i - width/2, ma + lo, f'{ma:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(i + width/2, mu + lo, f'{mu:.1f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('DMD (reimpl.) vs MER-MTL Configurations\n(Mean +/- Std over 4 seeds)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_dmd_vs_mermtl_grouped_bar.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_radar(results, output_dir):
    """Radar chart for aligned and unaligned."""
    models = [m for m in ['DMD', 'MER_MTL_TT', 'MER_MTL_MP'] if m in results]
    if len(models) < 2:
        return None

    radar_metrics = ['acc7', 'acc2', 'f1', 'mae', 'corr']
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        for model in models:
            values = []
            for metric in radar_metrics:
                m, s = _mean_std(results.get(model, {}).get(mode, {}).get(metric, {}))
                if m is None:
                    values.append(0)
                    continue
                if metric in ('acc7', 'acc2', 'f1'):
                    values.append(m / 100.0)
                elif metric == 'mae':
                    values.append(max(0, 1.0 - m))
                elif metric == 'corr':
                    values.append(m)
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2.5,
                    color=COLORS.get(model, '#95A5A6'),
                    label=DISPLAY_NAMES.get(model, model))
            ax.fill(angles, values, alpha=0.12, color=COLORS.get(model, '#95A5A6'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in radar_metrics], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{mode.capitalize()}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle('DMD (reimpl.) vs MER-MTL Radar Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_dmd_vs_mermtl_radar.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_per_metric_bar(results, output_dir):
    """Individual bar charts per metric."""
    models = [m for m in ['DMD', 'MER_MTL_TT', 'MER_MTL_MP'] if m in results]
    paths = {}

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(models))
        width = 0.35

        aligned_vals, unaligned_vals = [], []
        aligned_errs, unaligned_errs = [], []

        for model in models:
            m_a, s_a = _mean_std(results.get(model, {}).get('aligned', {}).get(metric, {}))
            m_u, s_u = _mean_std(results.get(model, {}).get('unaligned', {}).get(metric, {}))
            aligned_vals.append(m_a or 0)
            unaligned_vals.append(m_u or 0)
            aligned_errs.append(s_a or 0)
            unaligned_errs.append(s_u or 0)

        bars_a = ax.bar(x - width/2, aligned_vals, width, yerr=aligned_errs,
                        capsize=5, label='Aligned', color='#2980B9', edgecolor='white')
        bars_u = ax.bar(x + width/2, unaligned_vals, width, yerr=unaligned_errs,
                        capsize=5, label='Unaligned', color='#E67E22', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in models], fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(f'{METRIC_LABELS[metric]} - Model Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        all_v2 = aligned_vals + unaligned_vals
        ym2 = max(all_v2) if all_v2 else 1.0
        ax.set_ylim(0, ym2 * 1.2)

        # Proportional label offset based on data range
        all_v = aligned_vals + unaligned_vals
        ym = max(all_v) if all_v else 1.0
        lo = max(ym * 0.05, 0.02)
        for i, (ma, mu) in enumerate(zip(aligned_vals, unaligned_vals)):
            ax.text(i - width/2, ma + lo, f'{ma:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, mu + lo, f'{mu:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, f'fig_dmd_vs_mermtl_{metric}_bar.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")
        paths[metric] = path

    return paths


def plot_summary_table(results, output_dir):
    """Professional summary table."""
    models = [m for m in ['DMD', 'MER_MTL_TT', 'MER_MTL_MP'] if m in results]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    header = ['Model', 'Mode'] + [METRIC_LABELS[m] for m in METRICS]

    table_data = []
    for model in models:
        for mode in MODES:
            row = [DISPLAY_NAMES.get(model, model), mode.capitalize()]
            for metric in METRICS:
                m, s = _mean_std(results.get(model, {}).get(mode, {}).get(metric, {}))
                if m is not None:
                    unit = '%' if metric in ('acc7', 'acc2', 'f1') else ''
                    row.append(f'{m:.2f}{unit}\n(+/-{s:.2f})')
                else:
                    row.append('N/A')
            table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=header,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(len(header)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    color_map = {'DMD': '#FADBD8', 'MER_MTL_TT': '#D6EAF8', 'MER_MTL_MP': '#FDEBD0'}
    for i, row in enumerate(table_data):
        model_display = row[0]
        for model_key, color in color_map.items():
            if DISPLAY_NAMES.get(model_key, model_key) == model_display:
                for j in range(len(header)):
                    table[i + 1, j].set_facecolor(color)
                break

    fig.suptitle('Summary: DMD (reimpl.) vs MER-MTL Configurations\n(4 seeds, mean +/- std)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_dmd_vs_mermtl_summary_table.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_with_paper_baseline(results, output_dir):
    """Bar chart including DMD Paper baselines."""
    models = [m for m in ['MER_MTL_TT', 'MER_MTL_MP'] if m in results]
    if not models:
        return None

    all_models = models + ['DMD', 'DMD_Paper']

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 7))
    if len(METRICS) == 1:
        axes = [axes]

    x = np.arange(len(all_models))
    width = 0.35

    for ax, metric in zip(axes, METRICS):
        aligned_vals, unaligned_vals = [], []

        for model in all_models:
            if model == 'DMD_Paper':
                aligned_vals.append(DMD_PAPER['aligned'].get(metric, 0))
                unaligned_vals.append(DMD_PAPER['unaligned'].get(metric, 0))
            else:
                m_a, _ = _mean_std(results.get(model, {}).get('aligned', {}).get(metric, {}))
                m_u, _ = _mean_std(results.get(model, {}).get('unaligned', {}).get(metric, {}))
                aligned_vals.append(m_a or 0)
                unaligned_vals.append(m_u or 0)

        bars_a = ax.bar(x - width/2, aligned_vals, width, label='Aligned', edgecolor='white')
        bars_u = ax.bar(x + width/2, unaligned_vals, width, label='Unaligned',
                        edgecolor='white', hatch='//')

        for i, (ba, bu) in enumerate(zip(bars_a, bars_u)):
            c = COLORS.get(all_models[i], '#95A5A6')
            ba.set_facecolor(c)
            bu.set_facecolor(c)
            bu.set_alpha(0.6)
            bu.set_hatch('//')

        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in all_models],
                           fontsize=8, rotation=15, ha='right')
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        all_v2 = aligned_vals + unaligned_vals
        ym2 = max(all_v2) if all_v2 else 1.0
        ax.set_ylim(0, ym2 * 1.2)

        # Proportional label offset based on data range
        all_v = aligned_vals + unaligned_vals
        ym = max(all_v) if all_v else 1.0
        lo = max(ym * 0.05, 0.02)
        for i, (ma, mu) in enumerate(zip(aligned_vals, unaligned_vals)):
            ax.text(i - width/2, ma + lo, f'{ma:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(i + width/2, mu + lo, f'{mu:.1f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Our Results vs DMD Paper Baselines\n(Mean over 4 seeds)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_all_models_vs_paper.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='DMD (reimpl.) vs MER-MTL Comparison Visualization (v3)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Experiment results directory')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures')
    parser.add_argument('--log_dir', type=str, default='./logs/mermtl',
                        help='Training logs directory (default: ./logs/mermtl)')
    parser.add_argument('--no_curves', action='store_true',
                        help='Skip training curve plot')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("DMD (reimpl.) vs MER-MTL Comparison Visualization (v3)")
    print(f"{'=' * 60}\n")

    results = build_results(args)

    if not results:
        print("\nERROR: No results found!")
        print("Expected directory layout:")
        print(f"  {args.results_dir}/")
        print(f"    dmd/results_aligned.txt")
        print(f"    dmd/results_unaligned.txt")
        print(f"    mermtl/MER_MTL_{{text_mode}}_{{mode}}_seed{{N}}/metrics.json")
        sys.exit(1)

    # Print loaded data summary
    print(f"\n{'=' * 60}")
    print("Loaded data summary:")
    print(f"{'=' * 60}")
    for model in sorted(results):
        print(f"\n  {DISPLAY_NAMES.get(model, model)}:")
        for mode in MODES:
            for metric in METRICS:
                vals = results.get(model, {}).get(mode, {}).get(metric, {})
                if vals:
                    m, s = _mean_std(vals)
                    unit = '%' if metric in ('acc7', 'acc2', 'f1') else ''
                    print(f"    {mode:10s} {metric:5s}: {m:.2f}{unit} +/- {s:.2f}")

    print(f"\n{'=' * 60}")
    print("Generating visualizations...\n")

    plot_grouped_bar(results, args.output)
    plot_radar(results, args.output)
    plot_per_metric_bar(results, args.output)
    plot_summary_table(results, args.output)
    plot_with_paper_baseline(results, args.output)

    if not args.no_curves:
        print("\nGenerating training convergence curves...")
        plot_training_curves(logs_dir=args.log_dir, output_dir=args.output)

    print(f"\n{'=' * 60}")
    print(f"Done! All figures saved to: {args.output}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()

