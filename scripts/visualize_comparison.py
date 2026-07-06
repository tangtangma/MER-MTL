#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Visualization Script
Compare performance of DMD, MER-MTL-TT, MER-MTL-MP
in Aligned and Unaligned modes.
Supports mean comparison over 4 seeds (42, 1111, 1112, 1113).

Usage:
    # Demo mode (simulated data):
    python visualize_comparison.py --demo

    # From results directory (auto-detects DMD + MER-MTL results):
    python visualize_comparison.py --results_dir ./results

    # From CSV:
    python visualize_comparison.py --csv results_summary.csv
"""
import os
import sys
import json
import argparse
import csv
import re
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.visualization import generate_comparison_visualizations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = ['seed42', 'seed1111', 'seed1112', 'seed1113']
MODES = ['aligned', 'unaligned']
METRICS = ['acc7', 'acc2', 'f1', 'mae', 'corr']


def _empty_mode_dict():
    """Return an empty mode -> metric -> seed structure."""
    return {mode: {m: {} for m in METRICS} for mode in MODES}


# ---------------------------------------------------------------------------
# Loader: metrics.json (MER-MTL format)
# ---------------------------------------------------------------------------
def _load_metrics_json(metrics_path: str) -> dict:
    """Load a single metrics.json file, return normalized dict."""
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: could not read {metrics_path}: {e}")
        return None
    return data


# ---------------------------------------------------------------------------
# Loader: DMD txt results format
# ---------------------------------------------------------------------------
def _parse_dmd_txt(txt_path: str) -> dict:
    """
    Parse DMD results txt file.
    
    Supports formats like:
      - "Test Acc-7: 37.50%" or "Test Acc_7: 37.50%"
      - "Acc-7: 0.3750" (0-1 range)
      - "Best Test Acc-7: 37.50%"
      - "F1: 77.06" or "F1_score: 0.7706"
      - Per-seed lines: "seed42: Acc-7=37.50% Acc-2=79.65% F1=77.06 MAE=0.8955 Corr=0.6911"
      - Key=Value on separate lines
    
    Returns: {metric_key: value} where values are in the SAME scale as metrics.json
             (acc7/acc2/f1: 0-1, mae/corr: raw)
    """
    if not os.path.exists(txt_path):
        return None

    results = {}
    
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
    except IOError:
        return None

    lines = content.split('\n')
    
    # Pattern 1: "Best Test Acc-7: 37.50%" or "Test Acc-7: 37.50%"
    for line in lines:
        line_stripped = line.strip()
        
        # Try "Acc-7: XX.XX%" pattern
        m = re.search(r'(?:Best\s+)?Test\s+Acc[_-]?7\s*[:=]\s*([\d.]+)\s*%?', line_stripped, re.IGNORECASE)
        if m and 'acc7' not in results:
            val = float(m.group(1))
            # If > 1, it's percentage -> convert to 0-1
            if val > 1:
                val = val / 100.0
            results['acc7'] = val
            continue
        
        m = re.search(r'(?:Best\s+)?Test\s+Acc[_-]?2\s*[:=]\s*([\d.]+)\s*%?', line_stripped, re.IGNORECASE)
        if m and 'acc2' not in results:
            val = float(m.group(1))
            if val > 1:
                val = val / 100.0
            results['acc2'] = val
            continue
        
        # F1: could be percentage or 0-1
        m = re.search(r'(?:Best\s+)?Test\s+F1(?:_score)?\s*[:=]\s*([\d.]+)', line_stripped, re.IGNORECASE)
        if m and 'f1' not in results:
            val = float(m.group(1))
            if val > 1:
                val = val / 100.0
            results['f1'] = val
            continue
        
        m = re.search(r'(?:Best\s+)?Test\s+MAE\s*[:=]\s*([\d.]+)', line_stripped, re.IGNORECASE)
        if m and 'mae' not in results:
            results['mae'] = float(m.group(1))
            continue
        
        m = re.search(r'(?:Best\s+)?Test\s+Corr(?:elation)?\s*[:=]\s*([\d.-]+)', line_stripped, re.IGNORECASE)
        if m and 'corr' not in results:
            results['corr'] = float(m.group(1))
            continue

    # Pattern 2: compact per-seed lines
    # "seed42 Acc-7=37.50% Acc-2=79.65% F1=77.06 MAE=0.8955 Corr=0.6911"
    # (This is for per-seed parsing, but we use the overall "Best Test" values above)
    
    return results if results else None


def _parse_dmd_per_seed_results(txt_path: str) -> dict:
    """
    Parse DMD per-seed results from a txt file.
    
    Expected format in final_summary.txt or similar:
      Seed 42:   Acc-7=37.50% Acc-2=79.65% F1=77.06 MAE=0.8955 Corr=0.6911
      Seed 1111: Acc-7=38.10% Acc-2=80.10% F1=76.50 MAE=0.8900 Corr=0.6950
      ...
    
    Or:
      seed42:   acc7=0.3750 acc2=0.7965 f1=0.7706 mae=0.8955 corr=0.6911
    
    Returns: {'seed42': {metric: value}, 'seed1111': {...}, ...}
    """
    if not os.path.exists(txt_path):
        return None

    seed_results = {}
    
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
    except IOError:
        return None

    lines = content.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        # Match "seed42" or "Seed 42" at the beginning
        seed_match = re.match(r'(?:seed\s*|Seed\s+)?(\d+)\s*[:=\s]', line_stripped)
        if not seed_match:
            continue
        
        seed_num = seed_match.group(1)
        seed_key = f"seed{seed_num}"
        if seed_key not in SEEDS:
            continue
        
        metrics = {}
        
        # Parse Acc-7
        m = re.search(r'Acc[_-]?7\s*[=:]\s*([\d.]+)\s*%?', line_stripped, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            metrics['acc7'] = val / 100.0 if val > 1 else val
        
        # Parse Acc-2
        m = re.search(r'Acc[_-]?2\s*[=:]\s*([\d.]+)\s*%?', line_stripped, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            metrics['acc2'] = val / 100.0 if val > 1 else val
        
        # Parse F1
        m = re.search(r'F1(?:_score)?\s*[=:]\s*([\d.]+)', line_stripped, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            metrics['f1'] = val / 100.0 if val > 1 else val
        
        # Parse MAE
        m = re.search(r'MAE\s*[=:]\s*([\d.]+)', line_stripped, re.IGNORECASE)
        if m:
            metrics['mae'] = float(m.group(1))
        
        # Parse Corr
        m = re.search(r'Corr(?:elation)?\s*[=:]\s*([\d.-]+)', line_stripped, re.IGNORECASE)
        if m:
            metrics['corr'] = float(m.group(1))
        
        if metrics:
            seed_results[seed_key] = metrics
    
    return seed_results if seed_results else None


# ---------------------------------------------------------------------------
# Loader: results directory (unified)
# ---------------------------------------------------------------------------
def load_results_from_dir(base_dir: str) -> dict:
    """
    Walk the results directory and load results.
    
    Supports:
      1. metrics.json (MER-MTL format) - preferred
      2. DMD txt format (results_aligned.txt, final_summary.txt, etc.)
    
    Layout A (MER-MTL):
        results/mermtl/MER_MTL_{text_mode}_{mode}_seed{seed}/metrics.json
    
    Layout B (DMD):
        results/dmd/results_aligned.txt
        results/dmd/results_unaligned.txt
        results/dmd/final_summary.txt
    
    Returns:
        {model_key: {mode: {metric: {seed_key: value}}}}
        where model_key in ('DMD', 'MER_MTL_TT', 'MER_MTL_MP')
    """
    results = {}

    if not os.path.isdir(base_dir):
        print(f"Results directory not found: {base_dir}")
        return results

    # First pass: look for metrics.json files
    for root, dirs, files in os.walk(base_dir):
        if 'metrics.json' not in files:
            continue
        metrics_path = os.path.join(root, 'metrics.json')
        data = _load_metrics_json(metrics_path)
        if data is None:
            continue

        model_key = _infer_model_key(data, root)
        if model_key is None:
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
                # Store percentage for acc7/acc2/f1 (convert from 0-1 to 0-100)
                if metric in ('acc7', 'acc2', 'f1'):
                    results[model_key][mode][metric][seed_key] = float(val) * 100
                else:
                    results[model_key][mode][metric][seed_key] = float(val)

    # Second pass: look for DMD txt files (only if DMD not already loaded from metrics.json)
    if 'DMD' not in results:
        dmd_dir = os.path.join(base_dir, 'dmd')
        if os.path.isdir(dmd_dir):
            dmd_results = _load_dmd_txt_results(dmd_dir)
            if dmd_results:
                results['DMD'] = dmd_results
                print(f"  Loaded DMD results from txt files in {dmd_dir}")

    return results


def _load_dmd_txt_results(dmd_dir: str) -> dict:
    """
    Load DMD results from txt files.
    
    Tries multiple strategies:
    1. Per-seed per-mode files: results_aligned_seed42.txt, etc.
    2. Per-mode files: results_aligned.txt (with per-seed breakdown)
    3. Summary file: final_summary.txt (with per-seed breakdown)
    """
    dmd_results = _empty_mode_dict()
    found_any = False
    
    # Strategy 1: Try per-seed per-mode files
    for mode in MODES:
        for seed_key in SEEDS:
            seed_num = seed_key.replace('seed', '')
            fname = f"results_{mode}_{seed_num}.txt"
            fpath = os.path.join(dmd_dir, fname)
            if os.path.exists(fpath):
                parsed = _parse_dmd_txt(fpath)
                if parsed:
                    for metric, val in parsed.items():
                        if metric in ('acc7', 'acc2', 'f1'):
                            dmd_results[mode][metric][seed_key] = val * 100  # to percentage
                        else:
                            dmd_results[mode][metric][seed_key] = val
                    found_any = True
    
    if found_any:
        return dmd_results
    
    # Strategy 2: Try per-mode files (results_aligned.txt)
    for mode in MODES:
        for fname in [f"results_{mode}.txt", f"test_{mode}.txt"]:
            fpath = os.path.join(dmd_dir, fname)
            if os.path.exists(fpath):
                # First try per-seed parsing
                seed_data = _parse_dmd_per_seed_results(fpath)
                if seed_data:
                    for seed_key, metrics in seed_data.items():
                        for metric, val in metrics.items():
                            if metric in ('acc7', 'acc2', 'f1'):
                                dmd_results[mode][metric][seed_key] = val * 100
                            else:
                                dmd_results[mode][metric][seed_key] = val
                    found_any = True
                    break
                
                # Fall back to overall results (apply same value to all seeds)
                parsed = _parse_dmd_txt(fpath)
                if parsed:
                    for seed_key in SEEDS:
                        for metric, val in parsed.items():
                            if metric in ('acc7', 'acc2', 'f1'):
                                dmd_results[mode][metric][seed_key] = val * 100
                            else:
                                dmd_results[mode][metric][seed_key] = val
                    found_any = True
                    break
    
    if found_any:
        return dmd_results
    
    # Strategy 3: Try final_summary.txt
    summary_path = os.path.join(dmd_dir, 'final_summary.txt')
    if os.path.exists(summary_path):
        seed_data = _parse_dmd_per_seed_results(summary_path)
        if seed_data:
            for mode in MODES:
                for seed_key, metrics in seed_data.items():
                    for metric, val in metrics.items():
                        if metric in ('acc7', 'acc2', 'f1'):
                            dmd_results[mode][metric][seed_key] = val * 100
                        else:
                            dmd_results[mode][metric][seed_key] = val
            found_any = True
        else:
            parsed = _parse_dmd_txt(summary_path)
            if parsed:
                for mode in MODES:
                    for seed_key in SEEDS:
                        for metric, val in parsed.items():
                            if metric in ('acc7', 'acc2', 'f1'):
                                dmd_results[mode][metric][seed_key] = val * 100
                            else:
                                dmd_results[mode][metric][seed_key] = val
                found_any = True
    
    return dmd_results if found_any else None


def _infer_model_key(data: dict, dir_path: str) -> str:
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


def _infer_mode_from_path(dir_path: str) -> str:
    path_lower = dir_path.lower()
    if 'unaligned' in path_lower:
        return 'unaligned'
    if 'aligned' in path_lower:
        return 'aligned'
    return None


def _infer_seed_from_path(dir_path: str) -> str:
    m = re.search(r'seed(\d+)', dir_path, re.IGNORECASE)
    if m:
        return f"seed{m.group(1)}"
    return ''


# ---------------------------------------------------------------------------
# Loader: CSV
# ---------------------------------------------------------------------------
def load_results_from_csv(csv_path: str) -> dict:
    """Load results from a summary CSV.

    Expected columns: model, mode, seed, acc7, acc2, f1, mae, corr
    Values for acc7/acc2/f1: 0-1 range (will be converted to percentage).
    Values for mae/corr: raw.
    """
    results = {}
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return results

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get('model', '').strip()
            mode = row.get('mode', '').strip()
            seed = row.get('seed', '').strip()

            if not model or not mode or not seed:
                continue

            if not seed.startswith('seed'):
                seed = f"seed{seed}"

            # Normalize model key
            model_lower = model.lower()
            if 'mer_mtl_tt' in model_lower or 'mer-mtl-tt' in model_lower:
                model = 'MER_MTL_TT'
            elif 'mer_mtl_mp' in model_lower or 'mer-mtl-mp' in model_lower:
                model = 'MER_MTL_MP'
            elif 'dmd' in model_lower:
                model = 'DMD'

            if model not in results:
                results[model] = _empty_mode_dict()

            for metric in METRICS:
                val_str = row.get(metric, '').strip()
                if val_str:
                    val = float(val_str)
                    # Convert acc7/acc2/f1 from 0-1 to percentage
                    if metric in ('acc7', 'acc2', 'f1') and val <= 1.0:
                        val = val * 100
                    results[model][mode][metric][seed] = val

    return results


# ---------------------------------------------------------------------------
# Generate metrics.json for DMD from txt results
# ---------------------------------------------------------------------------
def generate_dmd_metrics_json(dmd_dir: str, results_dict: dict):
    """
    Generate metrics.json files for DMD results so they can be reused.
    Creates: results/dmd/{mode}/metrics.json for each seed.
    """
    if 'DMD' not in results_dict:
        return
    
    for mode in MODES:
        for seed_key in SEEDS:
            seed_num = int(seed_key.replace('seed', ''))
            
            # Check if there's any data for this mode+seed
            has_data = False
            metrics_data = {
                'model': 'DMD',
                'mode': mode,
                'seed': seed_num,
            }
            
            for metric in METRICS:
                val = results_dict['DMD'].get(mode, {}).get(metric, {}).get(seed_key)
                if val is not None:
                    has_data = True
                    # Convert back from percentage to 0-1 for storage
                    if metric in ('acc7', 'acc2', 'f1'):
                        metrics_data[metric] = val / 100.0
                    else:
                        metrics_data[metric] = val
            
            if has_data:
                out_dir = os.path.join(dmd_dir, f"results_{mode}_{seed_key}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, 'metrics.json')
                with open(out_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"  Generated: {out_path}")


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------
def _make_demo_results() -> dict:
    """Generate realistic simulated data for demo visualisation."""
    demo = {}
    demo['DMD'] = {
        'aligned': {
            'acc7': {'seed42': 41.2, 'seed1111': 40.8, 'seed1112': 41.5, 'seed1113': 40.5},
            'acc2': {'seed42': 83.1, 'seed1111': 82.7, 'seed1112': 83.4, 'seed1113': 82.9},
            'f1':   {'seed42': 42.5, 'seed1111': 42.1, 'seed1112': 43.0, 'seed1113': 42.3},
            'mae':  {'seed42': 0.78, 'seed1111': 0.79, 'seed1112': 0.77, 'seed1113': 0.80},
            'corr': {'seed42': 0.49, 'seed1111': 0.48, 'seed1112': 0.50, 'seed1113': 0.47},
        },
        'unaligned': {
            'acc7': {'seed42': 39.8, 'seed1111': 39.2, 'seed1112': 40.1, 'seed1113': 39.5},
            'acc2': {'seed42': 81.5, 'seed1111': 81.0, 'seed1112': 81.8, 'seed1113': 81.2},
            'f1':   {'seed42': 41.0, 'seed1111': 40.5, 'seed1112': 41.3, 'seed1113': 40.8},
            'mae':  {'seed42': 0.82, 'seed1111': 0.83, 'seed1112': 0.81, 'seed1113': 0.84},
            'corr': {'seed42': 0.45, 'seed1111': 0.44, 'seed1112': 0.46, 'seed1113': 0.43},
        },
    }
    demo['MER_MTL_TT'] = {
        'aligned': {
            'acc7': {'seed42': 43.88, 'seed1111': 43.1, 'seed1112': 42.8, 'seed1113': 43.5},
            'acc2': {'seed42': 78.72, 'seed1111': 79.8, 'seed1112': 79.5, 'seed1113': 80.0},
            'f1':   {'seed42': 43.26, 'seed1111': 43.4, 'seed1112': 43.1, 'seed1113': 43.7},
            'mae':  {'seed42': 0.76, 'seed1111': 0.77, 'seed1112': 0.78, 'seed1113': 0.75},
            'corr': {'seed42': 0.51, 'seed1111': 0.50, 'seed1112': 0.49, 'seed1113': 0.52},
        },
        'unaligned': {
            'acc7': {'seed42': 41.0, 'seed1111': 41.5, 'seed1112': 41.2, 'seed1113': 41.8},
            'acc2': {'seed42': 77.8, 'seed1111': 78.2, 'seed1112': 78.0, 'seed1113': 78.5},
            'f1':   {'seed42': 40.5, 'seed1111': 40.9, 'seed1112': 40.7, 'seed1113': 41.2},
            'mae':  {'seed42': 0.80, 'seed1111': 0.79, 'seed1112': 0.81, 'seed1113': 0.78},
            'corr': {'seed42': 0.47, 'seed1111': 0.48, 'seed1112': 0.46, 'seed1113': 0.49},
        },
    }
    demo['MER_MTL_MP'] = {
        'aligned': {
            'acc7': {'seed42': 42.8, 'seed1111': 43.3, 'seed1112': 43.0, 'seed1113': 43.7},
            'acc2': {'seed42': 79.5, 'seed1111': 80.0, 'seed1112': 79.7, 'seed1113': 80.3},
            'f1':   {'seed42': 42.0, 'seed1111': 42.6, 'seed1112': 42.3, 'seed1113': 43.0},
            'mae':  {'seed42': 0.77, 'seed1111': 0.76, 'seed1112': 0.77, 'seed1113': 0.75},
            'corr': {'seed42': 0.50, 'seed1111': 0.51, 'seed1112': 0.50, 'seed1113': 0.52},
        },
        'unaligned': {
            'acc7': {'seed42': 41.3, 'seed1111': 41.8, 'seed1112': 41.5, 'seed1113': 42.0},
            'acc2': {'seed42': 78.0, 'seed1111': 78.5, 'seed1112': 78.2, 'seed1113': 78.7},
            'f1':   {'seed42': 40.8, 'seed1111': 41.3, 'seed1112': 41.0, 'seed1113': 41.5},
            'mae':  {'seed42': 0.79, 'seed1111': 0.78, 'seed1112': 0.80, 'seed1113': 0.77},
            'corr': {'seed42': 0.48, 'seed1111': 0.49, 'seed1112': 0.47, 'seed1113': 0.50},
        },
    }
    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='MER-MTL Comprehensive Visualization: DMD vs MER-MTL-TT vs MER-MTL-MP')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Experiment results directory (default: ./results)')
    parser.add_argument('--csv', type=str, default='',
                        help='Load results from CSV file (takes priority)')
    parser.add_argument('--output', type=str, default='./figures',
                        help='Output directory for figures (default: ./figures)')
    parser.add_argument('--demo', action='store_true',
                        help='Generate demo visualization with sample data')
    parser.add_argument('--generate_dmd_json', action='store_true',
                        help='Generate metrics.json for DMD txt results (one-time conversion)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        print("\n" + "=" * 70)
        print("Generating Demo Visualization (Simulated Data)")
        print("=" * 70)
        results = _make_demo_results()
    elif args.csv and os.path.exists(args.csv):
        print(f"\nLoading results from CSV: {args.csv}")
        results = load_results_from_csv(args.csv)
    elif os.path.isdir(args.results_dir):
        print(f"\nLoading results from directory: {args.results_dir}")
        results = load_results_from_dir(args.results_dir)
    else:
        print(f"\nNo results found at '{args.results_dir}'.")
        print("Generating demo visualization instead...\n")
        results = _make_demo_results()

    if not results:
        print("ERROR: No valid results loaded. Check your data paths.")
        sys.exit(1)

    # Generate DMD metrics.json if requested
    if args.generate_dmd_json and 'DMD' in results:
        dmd_dir = os.path.join(args.results_dir, 'dmd')
        if os.path.isdir(dmd_dir):
            print("\nGenerating DMD metrics.json files...")
            generate_dmd_metrics_json(dmd_dir, results)

    # Print loaded summary
    print("\n" + "=" * 70)
    print("Loaded Results Summary")
    print("=" * 70)
    for model in sorted(results):
        print(f"\n  {model}:")
        for mode in MODES:
            for metric in METRICS:
                vals = list(results[model].get(mode, {}).get(metric, {}).values())
                if vals:
                    unit = '%' if metric in ('acc7', 'acc2', 'f1') else ''
                    print(f"    {mode:10s} {metric:5s}: "
                          f"mean={np.mean(vals):.4f}{unit}, std={np.std(vals):.4f}, "
                          f"n={len(vals)}")

    # Generate visualizations
    paths = generate_comparison_visualizations(results, args.output)

    print("\n" + "=" * 70)
    print("All visualizations saved!")
    print("=" * 70)
    for key, path in paths.items():
        if isinstance(path, dict):
            for k, v in path.items():
                print(f"  {key}/{k}: {v}")
        else:
            print(f"  {key}: {path}")
    print()


if __name__ == '__main__':
    main()

