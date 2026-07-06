#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Results Collection Script
Collect all experiment results, compute mean and variance over 4 seeds, generate summary tables
"""
import os
import sys
import json
import argparse
import csv
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SEEDS = ['seed42', 'seed1111', 'seed1112', 'seed1113']
MODES = ['aligned', 'unaligned']
METRICS = ['acc7', 'acc2', 'f1', 'mae', 'corr']


def load_single_result(metrics_path: str) -> dict:
    """Load single experiment's metrics.json"""
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def collect_results(results_dir: str) -> dict:
    """Collect all experiment results from results directory
    
    Directory structure:
    results/
    ├── DMD/
    │   ├── aligned_seed42/metrics.json
    │   ├── aligned_seed1111/metrics.json
    │   └── ...
    ├── MER_MTL_TT/
    │   └── ...
    └── MER_MTL_MP/
        └── ...
    """
    results = {}
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results
    
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        results[model_dir] = {}
        
        for mode in MODES:
            results[model_dir][mode] = {m: {} for m in METRICS}
        
        # Iterate through seed and mode directories
        for sub_dir in os.listdir(model_path):
            sub_path = os.path.join(model_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            
            # Parse directory name: mode_seedXXX
            parts = sub_dir.rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            mode_part = parts[0]
            seed_part = parts[1]
            
            if mode_part not in MODES or seed_part not in SEEDS:
                continue
            
            # Read metrics.json
            metrics = load_single_result(os.path.join(sub_path, 'metrics.json'))
            if metrics:
                for metric in METRICS:
                    value = metrics.get(metric, 0)
                    # If value is percentage form (>1), convert to decimal
                    if value > 1:
                        value = value / 100
                    results[model_dir][mode_part][metric][seed_part] = value
    
    return results


def compute_summary(results: dict) -> dict:
    """Compute mean and standard deviation for each model in each mode"""
    summary = {}
    
    for model in results:
        summary[model] = {}
        for mode in MODES:
            summary[model][mode] = {}
            for metric in METRICS:
                values = list(results[model][mode][metric].values())
                if values:
                    summary[model][mode][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
    
    return summary


def print_summary_table(summary: dict):
    """Print summary table"""
    print("\n" + "=" * 80)
    print("Experiment Results Summary (Mean ± Std over 4 seeds)")
    print("=" * 80)
    
    # Table header
    header = f"{'Model':<20} {'Mode':<12}"
    for m in METRICS:
        header += f" {m.upper():<15}"
    print(header)
    print("-" * 80)
    
    # Data rows
    for model in sorted(summary.keys()):
        for mode in MODES:
            row = f"{model:<20} {mode.capitalize():<12}"
            for metric in METRICS:
                if metric in summary[model][mode]:
                    data = summary[model][mode][metric]
                    row += f" {data['mean']*100:>6.2f}±{data['std']*100:<6.2f} "
                else:
                    row += f" {'N/A':<15}"
            print(row)
        print()


def print_detailed_table(results: dict):
    """Print detailed results for each seed"""
    print("\n" + "=" * 80)
    print("Detailed Experiment Results (Performance per seed)")
    print("=" * 80)
    
    for model in sorted(results.keys()):
        print(f"\n[{model}]")
        for mode in MODES:
            print(f"\n  {mode.capitalize()} mode:")
            header = f"    {'Metric':<10}"
            for seed in SEEDS:
                header += f" {seed.replace('seed', ''):<10}"
            header += f" {'Mean±Std':<15}"
            print(header)
            print("    " + "-" * 60)
            
            for metric in METRICS:
                values = results[model][mode][metric]
                row = f"    {metric.upper():<10}"
                for seed in SEEDS:
                    val = values.get(seed, None)
                    row += f" {val*100:>6.2f}    " if val is not None else f" {'N/A':<10}"
                
                if values:
                    mean = np.mean(list(values.values()))
                    std = np.std(list(values.values()))
                    row += f" {mean*100:>6.2f}±{std*100:<6.2f}"
                else:
                    row += f" {'N/A':<15}"
                print(row)


def save_csv(summary: dict, output_path: str):
    """Save summary results to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Model', 'Mode', 'Metric', 'Seed42', 'Seed1111', 'Seed1112', 'Seed1113', 'Mean', 'Std']
        writer.writerow(header)
        
        # Data rows
        for model in sorted(summary.keys()):
            for mode in MODES:
                for metric in METRICS:
                    if metric in summary[model][mode]:
                        data = summary[model][mode][metric]
                        values = data['values']
                        row = [
                            model,
                            mode,
                            metric,
                            values[0] * 100,
                            values[1] * 100,
                            values[2] * 100,
                            values[3] * 100,
                            data['mean'] * 100,
                            data['std'] * 100
                        ]
                        writer.writerow(row)


def save_summary_json(summary: dict, output_path: str):
    """Save summary results to JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='MER-MTL Experiment Results Collection')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Experiment results directory')
    parser.add_argument('--output', type=str, default='./results_summary.csv',
                       help='Output CSV file path')
    parser.add_argument('--json', type=str, default='',
                       help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed results per seed')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MER-MTL Experiment Results Collection")
    print(f"Results directory: {args.results_dir}")
    print("=" * 80)
    
    # Collect results
    results = collect_results(args.results_dir)
    
    if not results:
        print("\nNo experiment results found!")
        print(f"Please ensure the results directory exists with the following structure:")
        print(f"  {args.results_dir}/")
        print(f"    ├── DMD/")
        print(f"    │   ├── aligned_seed42/metrics.json")
        print(f"    │   └── ...")
        print(f"    ├── MER_MTL_TT/")
        print(f"    └── MER_MTL_MP/")
        return
    
    # Compute summary
    summary = compute_summary(results)
    
    # Print summary table
    print_summary_table(summary)
    
    # Print detailed table (optional)
    if args.verbose:
        print_detailed_table(results)
    
    # Save CSV
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    save_csv(summary, args.output)
    print(f"\nSummary table saved to: {args.output}")
    
    # Save JSON (optional)
    if args.json:
        save_summary_json(summary, args.json)
        print(f"JSON saved to: {args.json}")
    
    print("\n" + "=" * 80)
    print("Collection complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
