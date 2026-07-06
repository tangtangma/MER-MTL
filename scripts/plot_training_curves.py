#!/usr/bin/env python3
"""
MER-MTL Training Curves Visualization Script (fig02)
Generate 2x2 layout of training loss weight evolution curves, truncated to 30 epochs.

Usage:
    python plot_training_curves.py --logs_dir ./logs/mermtl --output_dir ./figures

Dependencies: numpy, matplotlib
"""
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- Constants ----------
AUX_TASK_NAMES = ['L_main', 'L_binary_happy', 'L_binary_sad', 'L_reg']
AUX_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
MAX_EPOCH = 30  # Truncate to 30 epochs

# ---------- Log Parsing ----------
def parse_log_sigmas(log_path):
    """Parse sigma values per epoch from training.log.
    Returns [{'epoch': int, 'sigmas': [float, ...]}, ...]
    Epochs beyond MAX_EPOCH are skipped.
    """
    data = []
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'Ep (\d+)/\d+.*sigmas=\[([^\]]+)\]', line)
            if m:
                epoch = int(m.group(1))
                if epoch > MAX_EPOCH:
                    continue
                sigmas = [float(s.strip()) for s in m.group(2).split(',')]
                data.append({'epoch': epoch, 'sigmas': sigmas})
    return data


def collect_log_paths(logs_dir):
    """Scan experiment directories under logs_dir, organized by mode x variant.
    Returns {'aligned'/'unaligned': {'MER-MTL-TT'/'MER-MTL-MP': [log_paths]}}
    """
    patterns = [('_tt_', 'MER-MTL-TT'), ('_mp_', 'MER-MTL-MP')]
    result = {}
    for exp_name in sorted(os.listdir(logs_dir)):
        exp_path = os.path.join(logs_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        log_file = os.path.join(exp_path, 'training.log')
        if not os.path.exists(log_file):
            continue
        # Determine mode
        if '_aligned_' in exp_name:
            mode = 'aligned'
        elif '_unaligned_' in exp_name:
            mode = 'unaligned'
        else:
            continue
        # Determine variant
        variant = None
        for substring, var_name in patterns:
            if substring in exp_name:
                variant = var_name
                break
        if variant is None:
            continue
        result.setdefault(mode, {}).setdefault(variant, []).append(log_file)
    return result


# ---------- Computation ----------
def compute_epoch_weights(log_paths):
    """For a given list of log files, compute mean and std of effective loss weights per epoch.
    Returns (epoch_weights_dict, num_tasks) or (None, None).
    """
    all_sigma_data = []
    for log_path in log_paths:
        data = parse_log_sigmas(log_path)
        if data:
            all_sigma_data.append(data)
    if not all_sigma_data:
        return None, None

    num_tasks = len(all_sigma_data[0][0]['sigmas'])
    epoch_weights = {ep: [[] for _ in range(num_tasks)] for ep in range(1, MAX_EPOCH + 1)}

    for sigma_data in all_sigma_data:
        for entry in sigma_data:
            ep = entry['epoch']
            sigmas = entry['sigmas']
            inv_sigma_sq = [1.0 / (s ** 2) for s in sigmas]
            total_inv = sum(inv_sigma_sq)
            eff_weights = [inv / total_inv for inv in inv_sigma_sq]
            for k in range(num_tasks):
                epoch_weights[ep][k].append(eff_weights[k])

    return epoch_weights, num_tasks


# ---------- Plotting ----------
def plot_training_curves(log_files_by_mode, output_dir):
    """Generate 2x2 training curves figure and save as PDF and PNG."""
    modes_grid = ['aligned', 'unaligned']
    variants = [('MER-MTL-TT', 'MER-MTL-TT'), ('MER-MTL-MP', 'MER-MTL-MP')]

    # Check data availability
    has_data = any(
        log_files_by_mode.get(mode, {}).get(var_key)
        for mode in modes_grid for _, var_key in variants
    )
    if not has_data:
        print("WARNING: No training log data available for curves figure.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, mode in enumerate(modes_grid):
        for col, (var_label, var_key) in enumerate(variants):
            ax = axes[row][col]
            log_paths = log_files_by_mode.get(mode, {}).get(var_key, [])
            epoch_weights, num_tasks = compute_epoch_weights(log_paths)

            if epoch_weights is None:
                ax.set_visible(False)
                continue

            task_names = AUX_TASK_NAMES[:num_tasks]
            x = list(range(1, MAX_EPOCH + 1))

            for k, name in enumerate(task_names):
                color = AUX_COLORS[k] if k < len(AUX_COLORS) else '#333333'
                means = [np.mean(epoch_weights[ep][k]) if epoch_weights[ep][k] else 0 for ep in x]
                stds  = [np.std(epoch_weights[ep][k])  if epoch_weights[ep][k] else 0 for ep in x]

                ax.plot(x, means, '-', color=color, linewidth=2.0, label=name, zorder=3)
                ax.fill_between(
                    x,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.12, color=color, zorder=2
                )

            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Eff. Loss Weight', fontsize=9, labelpad=6)
            ax.set_title(f'{mode.capitalize()} / {var_label}', fontsize=10, fontweight='bold')
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            # Legend positioned at center right to avoid curve overlap
            ax.legend(loc='upper left', fontsize=7.5, frameon=True,
                      edgecolor='gray', fancybox=False, facecolor='white',
                      framealpha=0.85)

    # Consistent border styling
    for ax in axes.flatten():
        if not ax.get_visible():
            continue
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    path_png = os.path.join(output_dir, 'fig_training_curves.png')
    path_pdf = os.path.join(output_dir, 'fig_training_curves.pdf')
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {path_png}")
    print(f"  Saved: {path_pdf}")
    return path_pdf


# ---------- Entry Point ----------
def main():
    parser = argparse.ArgumentParser(description='Generate MER-MTL training curves figure (fig02)')
    parser.add_argument('--logs_dir', type=str, default='./logs/mermtl',
                        help='Root directory of training logs')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for generated figures')
    args = parser.parse_args()

    print("=" * 60)
    print("Collecting training logs (truncated to 30 epochs)...")
    log_files_by_mode = collect_log_paths(args.logs_dir)
    for mode in ['aligned', 'unaligned']:
        for variant, paths in log_files_by_mode.get(mode, {}).items():
            print(f"  {mode}/{variant}: {len(paths)} logs")

    print("\nGenerating training curves figure...")
    plot_training_curves(log_files_by_mode, args.output_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
