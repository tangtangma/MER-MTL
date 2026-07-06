"""
MER-MTL Batch Run Script (2 x 2 x 4 = 16 experiments)
============================================================
Runs all combinations of:
  - Text mode: tt (Text Transformer), mp (Mean Pooling)
  - Data mode: aligned, unaligned
  - Seeds: 42, 1111, 1112, 1113

Total: 2 x 2 x 4 = 16 experiments

Directory structure:
    logs/mermtl/{exp_name}/training.log          - Training logs
    results/mermtl/{exp_name}/final_results.txt   - Human-readable results
    results/mermtl/{exp_name}/metrics.json        - Machine-readable results
    pt/mermtl/{save_name}.pth            - Model checkpoints

Usage:
    python scripts/run_mer_mtl.py
    python scripts/run_mer_mtl.py --epochs 30 --batch_size 16 --lr 0.0001
============================================================
"""
import os
import sys
import gc
import json
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Import the core experiment function directly (same process, tqdm works natively)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_single_experiment as _run_experiment


# =============================================================================
# CONFIGURATION
# =============================================================================
TEXT_MODES = ['tt', 'mp']
DATA_MODES = ['aligned', 'unaligned']
SEEDS = [42, 1111, 1112, 1113]

# Experiment settings
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 0.0001
DEFAULT_AUX_WEIGHT = 0.1


def setup_logger(log_file):
    """Setup logger"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('MMSA_BATCH')
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_experiment_and_collect(text_mode, mode, seed, args, logger):
    """
    Run a single experiment by calling main.run_single_experiment directly.
    Returns a result dict with metrics and metadata.
    """
    exp_name = f"MER_MTL_{text_mode}_{mode}_seed{seed}"

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Starting: {exp_name}")
    logger.info("=" * 70)

    start_time = time.time()
    try:
        final_test = _run_experiment(
            text_mode=text_mode,
            mode=mode,
            cls_mode='7cls',
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            aux_weight=args.aux_weight,
            log_dir=args.log_dir,
            results_dir=args.results_dir,
            checkpoint_dir=args.checkpoint_dir,
            use_uncertainty=not args.no_uncertainty,
        )

        elapsed = time.time() - start_time

        if final_test is not None:
            logger.info(f"Completed: {exp_name} ({elapsed:.1f}s)")
            # Load from metrics.json for full data
            results = load_metrics_json(args.results_dir, exp_name)
            if results is None:
                # Fallback: build from returned dict
                results = {
                    'Acc_7': final_test.get('Acc_7', 0),
                    'Acc_2': final_test.get('Acc_2', 0),
                    'F1_score': final_test.get('F1_score', 0),
                    'MAE': final_test.get('MAE', 0),
                    'Corr': final_test.get('Corr', 0),
                }
            success = True
        else:
            logger.error(f"Failed: {exp_name} (returned None)")
            success = False
            results = None
            elapsed = time.time() - start_time

    except Exception as e:
        logger.error(f"Error: {exp_name} - {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        success = False
        results = None
        elapsed = time.time() - start_time

    return {
        'text_mode': text_mode,
        'mode': mode,
        'seed': seed,
        'success': success,
        'elapsed': elapsed,
        'results': results,
        'exp_name': exp_name,
    }


def load_metrics_json(results_dir, exp_name):
    """Try to load metrics.json from results directory"""
    json_path = Path(results_dir) / exp_name / "metrics.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {
            'Acc_7': data.get('acc7', 0),
            'Acc_2': data.get('acc2', 0),
            'F1_score': data.get('f1', 0),           # binary F1 (DMD-aligned)
            'F1_weighted': data.get('f1_weighted', 0), # 7-class weighted F1
            'MAE': data.get('mae', 0),                # continuous MAE (DMD-aligned)
            'MAE_class': data.get('mae_class', 0),     # class-distance MAE
            'Corr': data.get('corr', 0),
        }
    except Exception:
        return None


def print_summary(all_results, logger):
    """Print summary table"""
    logger.info("")
    logger.info("=" * 90)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 90)

    logger.info(f"{'Exp Name':<40} {'Acc-7':>10} {'Acc-2':>10} {'F1':>10} {'MAE':>10} {'Corr':>10} {'Time':>10}")
    logger.info("-" * 100)

    for r in all_results:
        name = r['exp_name']
        if r['results']:
            acc7 = r['results'].get('Acc_7', 0)
            acc2 = r['results'].get('Acc_2', 0)
            f1 = r['results'].get('F1_score', 0)
            mae = r['results'].get('MAE', 0)
            corr = r['results'].get('Corr', 0)
        else:
            acc7 = acc2 = f1 = mae = corr = 0

        elapsed = f"{r['elapsed']:.0f}s"
        status = "" if r['success'] else " FAILED"
        logger.info(f"{name:<40} {acc7:>9.2%} {acc2:>9.2%} {f1:>9.2%} {mae:>9.4f} {corr:>9.4f} {elapsed:>10}{status}")

    logger.info("-" * 100)

    logger.info("")
    logger.info("AGGREGATE RESULTS (Mean +/- Std across 4 seeds)")
    logger.info("-" * 100)

    for text_mode in TEXT_MODES:
        for mode in DATA_MODES:
            key = f"{text_mode}_{mode}"
            subset = [r for r in all_results
                     if r['text_mode'] == text_mode and r['mode'] == mode and r['success']]

            if subset:
                acc7s = [r['results']['Acc_7'] for r in subset if r['results']]
                acc2s = [r['results']['Acc_2'] for r in subset if r['results']]
                f1s = [r['results']['F1_score'] for r in subset if r['results']]
                maes = [r['results']['MAE'] for r in subset if r['results']]
                corrs = [r['results']['Corr'] for r in subset if r['results']]

                if acc7s:
                    logger.info(f"{key:<20} | Acc-7: {np.mean(acc7s):.2%} +/- {np.std(acc7s):.2%} | "
                               f"Acc-2: {np.mean(acc2s):.2%} +/- {np.std(acc2s):.2%} | "
                               f"F1: {np.mean(f1s):.2%} +/- {np.std(f1s):.2%} | "
                               f"MAE: {np.mean(maes):.4f} +/- {np.std(maes):.4f} | "
                               f"Corr: {np.mean(corrs):.4f} +/- {np.std(corrs):.4f}")

    successful = [r for r in all_results if r['success']]
    if successful:
        all_acc7 = [r['results']['Acc_7'] for r in successful if r['results']]
        all_acc2 = [r['results']['Acc_2'] for r in successful if r['results']]
        all_f1 = [r['results']['F1_score'] for r in successful if r['results']]
        all_mae = [r['results']['MAE'] for r in successful if r['results']]
        all_corr = [r['results']['Corr'] for r in successful if r['results']]

        logger.info("-" * 100)
        logger.info(f"{'Overall':<20} | Acc-7: {np.mean(all_acc7):.2%} +/- {np.std(all_acc7):.2%} | "
                   f"Acc-2: {np.mean(all_acc2):.2%} +/- {np.std(all_acc2):.2%} | "
                   f"F1: {np.mean(all_f1):.2%} +/- {np.std(all_f1):.2%} | "
                   f"MAE: {np.mean(all_mae):.4f} +/- {np.std(all_mae):.4f} | "
                   f"Corr: {np.mean(all_corr):.4f} +/- {np.std(all_corr):.4f}")
        logger.info(f"Success rate: {len(successful)}/{len(all_results)}")

    logger.info("=" * 90)


def save_results_csv(all_results, output_path):
    """Save results to CSV"""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TextMode', 'DataMode', 'Seed', 'Acc_7', 'Acc_2', 'F1_score', 'F1_weighted', 'MAE', 'MAE_class', 'Corr', 'Success', 'Elapsed(s)'])

        for r in all_results:
            acc7 = r['results'].get('Acc_7', '') if r['results'] else ''
            acc2 = r['results'].get('Acc_2', '') if r['results'] else ''
            f1 = r['results'].get('F1_score', '') if r['results'] else ''
            f1w = r['results'].get('F1_weighted', '') if r['results'] else ''
            mae = r['results'].get('MAE', '') if r['results'] else ''
            mae_cls = r['results'].get('MAE_class', '') if r['results'] else ''
            corr = r['results'].get('Corr', '') if r['results'] else ''

            writer.writerow([
                r['text_mode'],
                r['mode'],
                r['seed'],
                acc7,
                acc2,
                f1,
                f1w,
                mae,
                mae_cls,
                corr,
                r['success'],
                f"{r['elapsed']:.0f}",
            ])

    return output_path


def main():
    parser = argparse.ArgumentParser(description='MER-MTL Batch Run')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                       help='Learning rate')
    parser.add_argument('--aux_weight', type=float, default=DEFAULT_AUX_WEIGHT,
                       help='Auxiliary task weight')
    parser.add_argument('--no_uncertainty', action='store_true',
                       help='Disable uncertainty weighting')
    parser.add_argument('--log_dir', type=str, default='./logs/mermtl',
                       help='Log directory')
    parser.add_argument('--results_dir', type=str, default='./results/mermtl',
                       help='Results directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./pt/mermtl',
                       help='Checkpoint directory')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip experiments with existing results')

    args = parser.parse_args()

    # Setup batch logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log = Path(args.log_dir) / f"batch_run_{timestamp}.log"
    logger = setup_logger(batch_log)

    total_configs = len(TEXT_MODES) * len(DATA_MODES) * len(SEEDS)

    logger.info("=" * 90)
    logger.info("MER-MTL BATCH RUN")
    logger.info("=" * 90)
    logger.info(f"Configurations: {len(TEXT_MODES)} text modes x {len(DATA_MODES)} data modes x {len(SEEDS)} seeds = {total_configs} experiments")
    logger.info(f"Text modes: {TEXT_MODES}")
    logger.info(f"Data modes: {DATA_MODES}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Aux Weight: {args.aux_weight}, Uncertainty: {not args.no_uncertainty}")
    logger.info(f"Logs: {args.log_dir}")
    logger.info(f"Results: {args.results_dir}")
    logger.info(f"Checkpoints: {args.checkpoint_dir}")
    logger.info("=" * 90)

    all_results = []
    total_runs = total_configs
    run_idx = 0

    for text_mode in TEXT_MODES:
        for mode in DATA_MODES:
            for seed in SEEDS:
                run_idx += 1

                if args.skip_existing:
                    exp_name = f"MER_MTL_{text_mode}_{mode}_seed{seed}"
                    result_file = Path(args.results_dir) / exp_name / "metrics.json"
                    if result_file.exists():
                        logger.info(f"[{run_idx}/{total_runs}] Skipping existing: {exp_name}")
                        continue

                result = run_experiment_and_collect(text_mode, mode, seed, args, logger)
                all_results.append(result)

                gc.collect()
                torch.cuda.empty_cache()

    print_summary(all_results, logger)

    csv_path = Path(args.results_dir) / f"batch_results_{timestamp}.csv"
    save_results_csv(all_results, csv_path)
    logger.info(f"\nResults saved to: {csv_path}")

    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()
