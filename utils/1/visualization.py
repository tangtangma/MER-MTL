"""
Visualization Module - Model Comparison
Supports DMD vs MER-MTL (TT/MP) comparison across aligned/unaligned modes.
Metrics: Acc-7, Acc-2, F1, MAE, Corr
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Academic color palette
COLORS = {
    'DMD':          '#3498db',
    'MER_MTL_TT':   '#e74c3c',
    'MER_MTL_MP':   '#2ecc71',
}
HATCHES = {
    'aligned':   '',
    'unaligned': '///',
}
METRIC_LABELS = {
    'acc7':  'Acc-7 (%)',
    'acc2':  'Acc-2 (%)',
    'f1':    'F1 Score (%)',
    'mae':   'MAE (lower is better)',
    'corr':  'Correlation',
}
PCT_METRICS = {'acc7', 'acc2', 'f1'}


class ComparisonPlotter:
    """Generate publication-quality comparison figures for DMD vs MER-MTL."""

    def __init__(self, base_dir: str = './figures'):
        self._base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_fig(self, fig, name: str) -> str:
        save_dir = os.path.join(self._base_dir, 'model_comparison')
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_stats(all_results: dict, metric: str):
        """Compute per-model per-mode mean/std over seeds."""
        models = sorted(all_results.keys())
        modes = ['aligned', 'unaligned']
        seeds = ['seed42', 'seed1111', 'seed1112', 'seed1113']

        stats = {}
        for model in models:
            stats[model] = {}
            for mode in modes:
                if mode not in all_results.get(model, {}):
                    continue
                if metric not in all_results[model].get(mode, {}):
                    continue
                vals = []
                for s in seeds:
                    v = all_results[model][mode][metric].get(s)
                    if v is not None:
                        vals.append(v)
                if vals:
                    stats[model][mode] = {
                        'mean': np.mean(vals),
                        'std': np.std(vals),
                        'values': vals,
                    }
        return stats, models, modes

    # ------------------------------------------------------------------
    # 1. Grouped bar chart (per metric, aligned vs unaligned)
    # ------------------------------------------------------------------
    def plot_grouped_bar(self, all_results: dict, metric: str = 'acc7',
                         show_std: bool = True) -> str:
        stats, models, modes = self._compute_stats(all_results, metric)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35

        aligned_means  = [stats[m].get('aligned', {}).get('mean', 0) for m in models]
        aligned_stds   = [stats[m].get('aligned', {}).get('std', 0) for m in models]
        unaligned_means = [stats[m].get('unaligned', {}).get('mean', 0) for m in models]
        unaligned_stds  = [stats[m].get('unaligned', {}).get('std', 0) for m in models]

        bars1 = ax.bar(x - width / 2, aligned_means, width,
                       yerr=aligned_stds if show_std else None,
                       label='Aligned', color='#3498db', capsize=4,
                       edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width / 2, unaligned_means, width,
                       yerr=unaligned_stds if show_std else None,
                       label='Unaligned', color='#e74c3c', capsize=4,
                       edgecolor='black', linewidth=0.8, hatch='///')

        ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper()), fontsize=12)
        ax.set_title(f'{METRIC_LABELS.get(metric, metric)}\n'
                     f'Mean over 4 seeds (42, 1111, 1112, 1113)', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Value labels on bars - improved positioning to avoid overlap
        # Compute y-axis limit first
        all_heights = []
        for m, s in zip(aligned_means, aligned_stds):
            all_heights.append(m + s)
        for m, s in zip(unaligned_means, unaligned_stds):
            all_heights.append(m + s)
        y_max = max(all_heights) if all_heights else 1.0
        label_offset = y_max * 0.05  # 5% of y-range as offset

        for bars, means, stds in [(bars1, aligned_means, aligned_stds),
                                   (bars2, unaligned_means, unaligned_stds)]:
            for bar, mean, std in zip(bars, means, stds):
                h = bar.get_height()
                if metric in PCT_METRICS:
                    txt = f'{mean:.1f}'
                    if show_std:
                        txt += f'\n(+/-{std:.1f})'
                else:
                    txt = f'{mean:.4f}'
                    if show_std:
                        txt += f'\n(+/-{std:.4f})'
                ax.text(bar.get_x() + bar.get_width() / 2., h + std + label_offset,
                        txt, ha='center', va='bottom', fontsize=7.5)

        # Add some headroom for labels
        ax.set_ylim(0, y_max * 1.25)

        plt.tight_layout()
        return self.save_fig(fig, f'comparison_{metric}_grouped_bar')

    # ------------------------------------------------------------------
    # 2. Heatmap
    # ------------------------------------------------------------------
    def plot_heatmap(self, all_results: dict, metric: str = 'acc7') -> str:
        stats, models, modes = self._compute_stats(all_results, metric)

        fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.8)))
        data = np.zeros((len(models), len(modes)))
        stds = np.zeros((len(models), len(modes)))
        for i, model in enumerate(models):
            for j, mode in enumerate(modes):
                data[i, j] = stats.get(model, {}).get(mode, {}).get('mean', 0)
                stds[i, j] = stats.get(model, {}).get(mode, {}).get('std', 0)

        cmap = 'RdYlGn' if metric in PCT_METRICS else 'RdYlGn_r'
        im = ax.imshow(data, cmap=cmap, aspect='auto')

        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m.capitalize() for m in modes], fontsize=12)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)

        # Show mean +/- std in heatmap cells
        for i in range(len(models)):
            for j in range(len(modes)):
                if metric in PCT_METRICS:
                    txt = f'{data[i, j]:.1f}\n+/-{stds[i, j]:.1f}'
                else:
                    txt = f'{data[i, j]:.4f}\n+/-{stds[i, j]:.4f}'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=11, fontweight='bold')

        ax.set_title(f'{METRIC_LABELS.get(metric, metric)} Heatmap\n(Mean +/- Std over 4 seeds)',
                     fontsize=13)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(METRIC_LABELS.get(metric, metric), fontsize=10)

        plt.tight_layout()
        return self.save_fig(fig, f'comparison_{metric}_heatmap')

    # ------------------------------------------------------------------
    # 3. Multi-metric combined bar chart (5 metrics side by side)
    # ------------------------------------------------------------------
    def plot_multi_metric(self, all_results: dict) -> str:
        metrics = ['acc7', 'acc2', 'f1', 'mae', 'corr']
        stats, models, modes = {}, sorted(all_results.keys()), ['aligned', 'unaligned']
        for m in metrics:
            stats[m], _, _ = self._compute_stats(all_results, m)

        fig, axes = plt.subplots(1, len(metrics), figsize=(24, 6))
        fig.suptitle('DMD vs MER-MTL (TT / MP) - Full Metric Comparison\n'
                     '(Mean +/- Std over 4 seeds)', fontsize=14, fontweight='bold')

        colors_mode = {'aligned': '#3498db', 'unaligned': '#e74c3c'}
        x = np.arange(len(models))
        width = 0.35

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for i, mode in enumerate(modes):
                means = [stats[metric].get(m, {}).get(mode, {}).get('mean', 0) for m in models]
                stds_val = [stats[metric].get(m, {}).get(mode, {}).get('std', 0) for m in models]
                offset = (i - 0.5) * width
                ax.bar(x + offset, means, width * 0.9, yerr=stds_val, capsize=3,
                       label=mode.capitalize(), color=colors_mode[mode],
                       alpha=0.85, edgecolor='black', linewidth=0.5,
                       hatch=HATCHES.get(mode, ''))

            ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
            ax.set_title(metric.upper(), fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('MER_MTL_', '') for m in models], fontsize=9)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        return self.save_fig(fig, 'comparison_all_metrics')

    # ------------------------------------------------------------------
    # 4. Radar chart (one per mode)
    # ------------------------------------------------------------------
    def plot_radar(self, all_results: dict) -> dict:
        """Radar chart comparing models on normalised metrics."""
        paths = {}
        metrics = ['acc7', 'acc2', 'f1', 'mae', 'corr']
        # Normalisation ranges for display
        # acc7/acc2/f1 are in percentage (0-100) after load_results_from_dir conversion
        norm_ranges = {'acc7': (0, 60), 'acc2': (0, 100), 'f1': (0, 100),
                       'mae': (0, 2), 'corr': (-1, 1)}

        for mode in ['aligned', 'unaligned']:
            stats_per_metric = {}
            for m in metrics:
                s, _, _ = self._compute_stats(all_results, m)
                stats_per_metric[m] = s

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # close the polygon

            fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]),
                              [m.upper() for m in metrics], fontsize=11)

            models = sorted(all_results.keys())
            for model in models:
                values = []
                for m in metrics:
                    mean_val = stats_per_metric[m].get(model, {}).get(mode, {}).get('mean', 0)
                    lo, hi = norm_ranges[m]
                    # For MAE lower is better -> invert
                    if m == 'mae':
                        norm = (hi - mean_val) / (hi - lo)
                    else:
                        norm = (mean_val - lo) / (hi - lo)
                    values.append(max(0, min(1, norm)))
                values += values[:1]

                color = COLORS.get(model, '#888888')
                ax.plot(angles, values, 'o-', linewidth=2,
                        label=model, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)

            ax.set_ylim(0, 1)
            ax.set_title(f'Model Comparison Radar ({mode.capitalize()})',
                         fontsize=14, fontweight='bold', pad=20)
            # Place legend below the chart to avoid clipping
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                      ncol=3, fontsize=10, frameon=True)

            plt.tight_layout()
            paths[mode] = self.save_fig(fig, f'comparison_radar_{mode}')

        return paths

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    @staticmethod
    def generate_summary_table(all_results: dict) -> str:
        models = sorted(all_results.keys())
        modes = ['aligned', 'unaligned']
        metrics = ['acc7', 'acc2', 'f1', 'mae', 'corr']
        seeds = ['seed42', 'seed1111', 'seed1112', 'seed1113']

        header = "| Model | Mode | " + " | ".join([m.upper() for m in metrics]) + " |"
        sep = "|---|---|" + "|".join(["---" for _ in metrics]) + "|"

        rows = []
        for model in models:
            for mode in modes:
                cells = [model, mode.capitalize()]
                for metric in metrics:
                    vals = []
                    for s in seeds:
                        v = all_results.get(model, {}).get(mode, {}).get(metric, {}).get(s)
                        if v is not None:
                            vals.append(v)
                    if vals:
                        mean = np.mean(vals)
                        std = np.std(vals)
                        if metric in PCT_METRICS:
                            cells.append(f"{mean:.2f}+-{std:.2f}")
                        elif metric == 'mae':
                            cells.append(f"{mean:.4f}+-{std:.4f}")
                        else:
                            cells.append(f"{mean:.4f}+-{std:.4f}")
                    else:
                        cells.append("N/A")
                rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header, sep] + rows)

    # ------------------------------------------------------------------
    # 6. Paper baseline comparison bar chart
    # ------------------------------------------------------------------
    def plot_paper_comparison(self, all_results: dict,
                              paper_results: dict = None) -> str:
        """
        Bar chart comparing our results with paper baselines.
        
        paper_results format: {
            'DMD (Paper)': {'aligned': {'acc7': 41.4, 'acc2': 84.7, 'f1': 84.3, 'mae': 1.156, 'corr': 0.704},
                           'unaligned': {'acc7': 40.8, 'acc2': 83.9, 'f1': 83.5, 'mae': 1.177, 'corr': 0.695}}
        }
        """
        if paper_results is None:
            # Default DMD paper baselines
            paper_results = {
                'DMD (Paper)': {
                    'aligned':   {'acc7': 41.4, 'acc2': 84.7, 'f1': 84.3, 'mae': 1.156, 'corr': 0.704},
                    'unaligned': {'acc7': 40.8, 'acc2': 83.9, 'f1': 83.5, 'mae': 1.177, 'corr': 0.695},
                }
            }

        metrics = ['acc7', 'acc2', 'f1', 'mae', 'corr']
        
        # Merge: our models + paper baselines
        all_model_keys = sorted(all_results.keys())
        paper_model_keys = sorted(paper_results.keys())

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))
        fig.suptitle('Our Results vs DMD Paper Baselines\n(Mean over 4 seeds)',
                     fontsize=14, fontweight='bold')

        for ax_idx, mode in enumerate(['aligned', 'unaligned']):
            ax = axes[ax_idx]
            
            # Collect all model names and their values
            model_names = []
            acc7_vals, acc2_vals, f1_vals = [], [], []
            mae_vals, corr_vals = [], []
            is_paper = []

            # Paper results first
            for pm in paper_model_keys:
                if mode in paper_results[pm]:
                    model_names.append(pm)
                    d = paper_results[pm][mode]
                    acc7_vals.append(d.get('acc7', 0))
                    acc2_vals.append(d.get('acc2', 0))
                    f1_vals.append(d.get('f1', 0))
                    mae_vals.append(d.get('mae', 0))
                    corr_vals.append(d.get('corr', 0))
                    is_paper.append(True)

            # Our results
            for m in all_model_keys:
                stats_acc7 = self._compute_stats(all_results, 'acc7')
                stats_acc2 = self._compute_stats(all_results, 'acc2')
                stats_f1 = self._compute_stats(all_results, 'f1')
                stats_mae = self._compute_stats(all_results, 'mae')
                stats_corr = self._compute_stats(all_results, 'corr')
                
                if mode in stats_acc7[0].get(m, {}):
                    model_names.append(m)
                    acc7_vals.append(stats_acc7[0][m][mode]['mean'])
                    acc2_vals.append(stats_acc2[0][m][mode]['mean'])
                    f1_vals.append(stats_f1[0][m][mode]['mean'])
                    mae_vals.append(stats_mae[0][m][mode]['mean'])
                    corr_vals.append(stats_corr[0][m][mode]['mean'])
                    is_paper.append(False)

            if not model_names:
                ax.set_title(f'{mode.capitalize()} - No data')
                continue

            x = np.arange(len(model_names))
            width = 0.15
            
            # Color: paper models in gray, ours in color
            bar_colors = []
            for i, ip in enumerate(is_paper):
                if ip:
                    bar_colors.append('#95a5a6')  # gray for paper
                else:
                    bar_colors.append(COLORS.get(model_names[i], '#3498db'))

            ax.bar(x - 2*width, acc7_vals, width, label='Acc-7', color='#2c3e50', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.bar(x - width, acc2_vals, width, label='Acc-2', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.bar(x, f1_vals, width, label='F1', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.bar(x + width, mae_vals, width, label='MAE', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.bar(x + 2*width, corr_vals, width, label='Corr', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)

            ax.set_title(f'{mode.capitalize()}', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha='right')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

            # Mark paper results with hatching
            for i, ip in enumerate(is_paper):
                if ip:
                    for offset in range(-2, 3):
                        for bar in ax.containers:
                            if bar[i] and hasattr(bar[i], 'set_hatch'):
                                bar[i].set_hatch('///')
                                bar[i].set_alpha(0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return self.save_fig(fig, 'comparison_paper_baselines')


def generate_comparison_visualizations(results_dict: dict,
                                       output_dir: str = './figures',
                                       paper_results: dict = None) -> dict:
    """Generate all comparison visualizations.

    Args:
        results_dict: {
            'DMD':        {'aligned': {'acc7': {'seed42': 37.5, ...}, ...}, ...},
            'MER_MTL_TT': {...},
            'MER_MTL_MP': {...}
        }
        (acc7/acc2/f1 should be in percentage 0-100, mae/corr in raw values)
        
        output_dir: Directory to save figures
        paper_results: Optional paper baseline results for comparison

    Returns:
        dict of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plotter = ComparisonPlotter(output_dir)

    print("Generating comparison visualizations...")
    paths = {}

    # Per-metric grouped bar + heatmap
    for metric in ['acc7', 'acc2', 'f1', 'mae', 'corr']:
        paths[metric] = {
            'grouped_bar': plotter.plot_grouped_bar(results_dict, metric=metric),
            'heatmap':     plotter.plot_heatmap(results_dict, metric=metric),
        }
        print(f"  done {metric.upper()}")

    # Multi-metric combined
    paths['all_metrics'] = plotter.plot_multi_metric(results_dict)
    print("  done all-metrics combined")

    # Radar
    paths['radar'] = plotter.plot_radar(results_dict)
    print("  done radar charts")

    # Paper baseline comparison
    paths['paper'] = plotter.plot_paper_comparison(results_dict, paper_results)
    print("  done paper baseline comparison")

    # Summary table
    summary = ComparisonPlotter.generate_summary_table(results_dict)
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary)

    return paths

# ---------------------------------------------------------------------------
# 7. Training Convergence Curves (from training.log files)
# ---------------------------------------------------------------------------
import re as _re

LOG_EPOCH_RE = _re.compile(
    r'Ep\s+(\d+)/(\d+)\s*\|\s*Best:(\d+)\s*\|\s*'
    r'VAL\s+Acc7=([\d.]+)%?\s+Acc2=([\d.]+)%?\s*\|\s*'
    r'TEST\s+Acc7=([\d.]+)%?\s*\|',
    _re.IGNORECASE,
)

LOG_SIGMAS_RE = _re.compile(
    r'sigmas\s*=\s*\[([\d.,\s]+)\]',
    _re.IGNORECASE,
)

LOG_TRAIN_RE = _re.compile(
    r'TRAIN\s*>>.*?Loss=([\d.]+)',
    _re.IGNORECASE,
)


def _parse_training_log(log_path):
    """Parse a single training.log and return per-epoch dicts."""
    if not os.path.isfile(log_path):
        return None
    epochs = []
    with open(log_path, 'r') as f:
        for line in f:
            m = LOG_EPOCH_RE.search(line)
            if m:
                epochs.append({
                    'epoch':  int(m.group(1)),
                    'val_acc7': float(m.group(4)),
                    'val_acc2': float(m.group(5)),
                    'test_acc7': float(m.group(6)),
                })
    return epochs if epochs else None


def _collect_log_paths(logs_dir, exp_name_patterns):
    """Walk logs_dir and find matching training.log files."""
    log_files = {}
    for root, dirs, files in os.walk(logs_dir):
        if 'training.log' not in files:
            continue
        log_path = os.path.join(root, 'training.log')
        dir_name = os.path.basename(root)
        for pattern, key in exp_name_patterns:
            if pattern.lower() in dir_name.lower():
                if key not in log_files:
                    log_files[key] = []
                log_files[key].append(log_path)
                break
    return log_files


def _parse_log_sigmas(log_path):
    """Parse training.log and return per-epoch sigma values."""
    if not os.path.isfile(log_path):
        return None
    epochs = []
    with open(log_path, 'r') as f:
        for line in f:
            # Extract epoch number
            m_ep = _re.search(r'Ep\s+(\d+)/', line)
            # Extract sigmas
            m_sig = LOG_SIGMAS_RE.search(line)
            if m_ep and m_sig:
                sigma_vals = [float(x.strip()) for x in m_sig.group(1).split(',') if x.strip()]
                epochs.append({
                    'epoch':  int(m_ep.group(1)),
                    'sigmas': sigma_vals,
                })
    return epochs if epochs else None


AUX_TASK_NAMES = ['L_rec', 'L_cyc', 'L_mar', 'L_ort']
AUX_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # red, blue, green, orange


def plot_training_curves(logs_dir='./logs/mermtl',
                         output_dir='./figures',
                         metric='val_acc7',
                         fig_size=(10, 5)):
    """
    Plot loss weight dynamics: how adaptive uncertainty weights evolve
    during training for MER-MTL-TT and MER-MTL-MP.

    Effective weight for task k = 0.5 / sigma_k^2 * aux_weight

    Args:
        logs_dir:   Path to logs/mermtl/
        output_dir: Directory to save figure
        metric:     unused (kept for API compat)
        fig_size:   (width, height) tuple

    Returns:
        str: saved figure path
    """
    AUX_WEIGHT = 0.1  # default aux_task_weight

    patterns = [
        ('_tt_',  'MER-MTL-TT'),
        ('_mp_',  'MER-MTL-MP'),
    ]
    log_files = _collect_log_paths(logs_dir, patterns)

    if not log_files:
        print(f'  WARNING: No training logs found in {logs_dir}')
        return None

    # Use subplots: one per variant (TT, MP), side by side
    variants = [(k, v) for k, v in patterns if log_files.get(v)]
    if not variants:
        print(f'  WARNING: No matching logs found')
        return None

    fig, axes = plt.subplots(1, len(variants), figsize=fig_size,
                             sharex=True, sharey=True)
    if len(variants) == 1:
        axes = [axes]

    for ax, (pat, variant) in zip(axes, variants):
        logs = log_files.get(variant, [])
        all_sigma_data = []
        for log_path in logs:
            data = _parse_log_sigmas(log_path)
            if data:
                all_sigma_data.append(data)

        if not all_sigma_data:
            continue

        # Determine number of aux tasks from first log
        num_tasks = len(all_sigma_data[0][0]['sigmas'])
        task_names = AUX_TASK_NAMES[:num_tasks]
        task_colors = AUX_COLORS[:num_tasks]

        # Compute effective weights per epoch, aggregate across seeds
        max_epoch = max(max(d['epoch'] for d in sd) for sd in all_sigma_data)
        epoch_weights = {ep: [[] for _ in range(num_tasks)] for ep in range(1, max_epoch + 1)}

        for sd in all_sigma_data:
            for item in sd:
                ep = item['epoch']
                sigmas = item['sigmas']
                for k in range(min(num_tasks, len(sigmas))):
                    sigma_k = sigmas[k]
                    w_k = 0.5 / (sigma_k ** 2) * AUX_WEIGHT
                    epoch_weights[ep][k].append(w_k)

        x = list(range(1, max_epoch + 1))
        for k, (name, color) in enumerate(zip(task_names, task_colors)):
            means = [np.mean(epoch_weights[ep][k]) if epoch_weights[ep][k] else 0 for ep in x]
            stds  = [np.std(epoch_weights[ep][k])  if epoch_weights[ep][k] else 0 for ep in x]

            ax.plot(x, means, '-', color=color, linewidth=2, label=name)
            ax.fill_between(x,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=color)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Effective Loss Weight', fontsize=11)
        ax.set_title(variant, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Adaptive Loss Weight Dynamics (Uncertainty Weighting)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'fig_training_curves.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path
