"""
Per-class Accuracy Extraction for DMD Baseline
===============================================
Loads a trained DMD checkpoint and computes per-class 7-class accuracy
on the test split. Output format matches extract_per_class_accuracy.py.

Usage:
    python extract_dmd_per_class.py \
        --checkpoint ./pt/dmd/dmd_aligned_seed42.pth \
        --data ./data/aligned_50.pkl \
        --mode aligned \
        --output ./results/per_class/dmd_aligned_seed42.json
"""
import argparse
import json
import os
import sys
import pickle
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import DMD model from run_dmd.py
from run_dmd import DMDModel, MMDataset


EMOTION_LABELS = {
    0: "Very Negative",
    1: "Negative",
    2: "Somewhat Neg",
    3: "Neutral",
    4: "Somewhat Pos",
    5: "Positive",
    6: "Very Positive",
}


def continuous_to_7class(labels):
    """Convert continuous sentiment labels [-3, +3] to 7-class indices."""
    boundaries = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    indices = np.digitize(labels, boundaries)
    indices = np.clip(indices - 1, 0, 6)
    return indices


def make_dmd_args(mode='aligned'):
    """Create args namespace matching DMD's expected configuration."""
    return Namespace(
        dataset_name='mosi',
        model_name='dmd',
        featurePath=f'./data/{mode}_50.pkl',
        train_mode='regression',
        KeyEval='Acc_7',
        need_data_aligned=(mode == 'aligned'),
        feature_dims=[768, 5, 20],
        dst_feature_dim_nheads=[50, 10],
        conv1d_kernel_size_l=5,
        conv1d_kernel_size_a=5,
        conv1d_kernel_size_v=5,
        attn_dropout=0.3,
        attn_dropout_a=0.2,
        attn_dropout_v=0.0,
        output_dropout=0.5,
        relu_dropout=0.0,
        res_dropout=0.0,
        text_dropout=0.5,
        embed_dropout=0.2,
        nlevels=4,
        attn_mask=False,
    )


def load_dmd_model(checkpoint_path, args, device='cpu'):
    """Load DMD model from checkpoint."""
    model = DMDModel(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def extract_per_class_dmd(model, dataloader, device='cpu'):
    """Extract per-class accuracy from DMD model predictions."""
    all_preds_cont = []
    all_labels_cont = []

    with torch.no_grad():
        for bd in dataloader:
            T = bd['text'].to(device)
            A = bd['audio'].to(device)
            V = bd['vision'].to(device)
            L = bd['labels']['M'].to(device).view(-1, 1)

            out = model(T, A, V, is_distill=True)
            all_preds_cont.append(out['output_logit'].cpu().numpy().flatten())
            all_labels_cont.append(L.cpu().numpy().flatten())

    preds_cont = np.concatenate(all_preds_cont)
    labels_cont = np.concatenate(all_labels_cont)

    # Convert continuous to 7-class (same as DMD evaluation)
    pred_7 = continuous_to_7class(np.clip(np.round(preds_cont), -3, 3))
    true_7 = continuous_to_7class(labels_cont)

    # Per-class accuracy
    per_class_acc = {}
    per_class_count = {}
    per_class_correct = {}

    for cls_idx in range(7):
        mask = (true_7 == cls_idx)
        count = int(mask.sum())
        if count > 0:
            correct = int((pred_7[mask] == cls_idx).sum())
            per_class_acc[str(cls_idx)] = round(correct / count, 4)
            per_class_count[str(cls_idx)] = count
            per_class_correct[str(cls_idx)] = correct
        else:
            per_class_acc[str(cls_idx)] = 0.0
            per_class_count[str(cls_idx)] = 0
            per_class_correct[str(cls_idx)] = 0

    # Confusion matrix
    confusion = np.zeros((7, 7), dtype=int)
    for t, p in zip(true_7, pred_7):
        confusion[int(t)][int(p)] += 1

    overall_acc = float((pred_7 == true_7).sum()) / len(true_7)

    return {
        'per_class_acc': per_class_acc,
        'per_class_count': per_class_count,
        'per_class_correct': per_class_correct,
        'confusion_matrix': confusion.tolist(),
        'overall_acc': round(overall_acc, 4),
        'emotion_labels': {str(k): v for k, v in EMOTION_LABELS.items()},
    }


def main():
    parser = argparse.ArgumentParser(description='Extract per-class accuracy for DMD')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='./data/aligned_50.pkl')
    parser.add_argument('--mode', type=str, default='aligned', choices=['aligned', 'unaligned'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print(f"Loading DMD model from: {args.checkpoint}")
    dmd_args = make_dmd_args(mode=args.mode)
    dmd_args.featurePath = args.data
    model = load_dmd_model(args.checkpoint, dmd_args, device=args.device)

    print(f"Loading test data from: {args.data}")
    ds_args = Namespace(featurePath=args.data)
    test_ds = MMDataset(ds_args, mode='test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Extracting per-class accuracy (device={args.device})...")
    results = extract_per_class_dmd(model, test_loader, device=args.device)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Per-class Accuracy (DMD): {args.mode}")
    print(f"{'='*60}")
    print(f"Overall Acc-7: {results['overall_acc']:.2%}")
    print(f"{'-'*60}")
    for cls_idx in range(7):
        label = EMOTION_LABELS[cls_idx]
        acc = results['per_class_acc'][str(cls_idx)]
        count = results['per_class_count'][str(cls_idx)]
        correct = results['per_class_correct'][str(cls_idx)]
        print(f"  Class {cls_idx} ({label:>15s}): {acc:.2%} ({correct}/{count})")
    print(f"{'='*60}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == '__main__':
    main()
