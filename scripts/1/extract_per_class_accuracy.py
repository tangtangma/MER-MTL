"""
Per-class Accuracy Extraction for MER-MTL
==========================================
Loads a trained MER-MTL checkpoint and computes per-class accuracy
for 7-class emotion recognition on test set.

Usage:
    python extract_per_class_accuracy.py \
        --checkpoint ./pt/mermtl/mer_mtl_aligned_mp_7cls.pth \
        --data ./data/aligned_50.pkl \
        --text_mode mp \
        --mode aligned \
        --output ./results/per_class/aligned_mp.json

Output: JSON file with per-class accuracy and confusion matrix.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# MOSI 7-class label mapping
# =============================================================================
# CMU-MOSI sentiment labels are continuous [-1, 1].
# Standard 7-class binning (used in most MER papers):
#   Class 0: [-1.0, -0.7)  -> Very Negative
#   Class 1: [-0.7, -0.4)  -> Negative
#   Class 2: [-0.4, -0.1)  -> Somewhat Negative
#   Class 3: [-0.1, 0.1)   -> Neutral
#   Class 4: [0.1, 0.4)    -> Somewhat Positive
#   Class 5: [0.4, 0.7)    -> Positive
#   Class 6: [0.7, 1.0]    -> Very Positive

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
    """Convert continuous sentiment labels to 7-class indices."""
    # Standard binning for CMU-MOSI 7-class
    bins = [-1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0]
    indices = np.digitize(labels, bins) - 1  # digitize returns 1-indexed
    indices = np.clip(indices, 0, 6)
    return indices


# =============================================================================
# Model loading (import from project)
# =============================================================================
def load_model(checkpoint_path, args):
    """Load MER-MTL model from checkpoint."""
    # Import model class
    # Add project root to path so 'models.mer_mtl_model' is importable
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    try:
        from models.mer_mtl_model import MERMTLModel
    except ImportError:
        # Fallback: if script is placed inside models/ directory
        from mer_mtl_model import MERMTLModel

    model = MERMTLModel(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_data(data_path, mode='aligned'):
    """Load dataset from pkl file."""
    # Try the given path first, then fallback to default location
    import glob
    if not os.path.exists(data_path):
        candidates = glob.glob(os.path.join(os.path.dirname(__file__), f'data/*{mode}*.pkl'))
        if candidates:
            data_path = candidates[0]
    print(f"  Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    # Handle different pkl structures
    if isinstance(data, dict) and 'test' in data:
        return data['test']
    elif isinstance(data, (list, tuple)) and len(data) >= 3:
        return data[2]  # typically (train, valid, test)
    else:
        return data


def prepare_batch(batch, device='cpu'):
    """Prepare a batch for model input."""
    T, A, V, L = batch[0], batch[1], batch[2], batch[3]
    text_mask = batch[4] if len(batch) > 4 else None
    audio_mask = batch[5] if len(batch) > 5 else None
    vision_mask = batch[6] if len(batch) > 6 else None

    T = T.float().to(device)
    A = A.float().to(device)
    V = V.float().to(device)
    L = L.long().to(device)

    if text_mask is not None:
        text_mask = text_mask.bool().to(device)
    if audio_mask is not None:
        audio_mask = audio_mask.bool().to(device)
    if vision_mask is not None:
        vision_mask = vision_mask.bool().to(device)

    return T, A, V, L, text_mask, audio_mask, vision_mask


def extract_per_class(model, test_data, device='cpu', batch_size=32):
    """
    Extract per-class predictions and compute per-class accuracy.

    Returns:
        dict with:
            - per_class_acc: {class_idx: accuracy}
            - per_class_count: {class_idx: count}
            - per_class_correct: {class_idx: correct_count}
            - confusion_matrix: 7x7 matrix
            - overall_acc: float
            - predictions: list of (pred_class, true_class, true_label_continuous)
    """
    # Prepare tensors
    text = torch.tensor(test_data['text'], dtype=torch.float32)
    audio = torch.tensor(test_data['audio'], dtype=torch.float32)
    vision = torch.tensor(test_data['vision'], dtype=torch.float32)
    labels = torch.tensor(test_data['labels'], dtype=torch.float32)

    # Handle masks if present
    has_masks = 'text_mask' in test_data or 'audio_mask' in test_data or 'vision_mask' in test_data
    if has_masks:
        text_mask = torch.tensor(test_data.get('text_mask', None), dtype=torch.bool) if 'text_mask' in test_data else None
        audio_mask = torch.tensor(test_data.get('audio_mask', None), dtype=torch.bool) if 'audio_mask' in test_data else None
        vision_mask = torch.tensor(test_data.get('vision_mask', None), dtype=torch.bool) if 'vision_mask' in test_data else None
        dataset = TensorDataset(text, audio, vision, labels,
                                text_mask if text_mask is not None else torch.zeros(text.shape[0], text.shape[1], dtype=torch.bool),
                                audio_mask if audio_mask is not None else torch.zeros(audio.shape[0], audio.shape[1], dtype=torch.bool),
                                vision_mask if vision_mask is not None else torch.zeros(vision.shape[0], vision.shape[1], dtype=torch.bool))
    else:
        dataset = TensorDataset(text, audio, vision, labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_trues = []
    all_true_continuous = []

    with torch.no_grad():
        for batch in dataloader:
            T, A, V, L, text_mask, audio_mask, vision_mask = prepare_batch(batch, device)
            out = model(T, A, V, text_mask, audio_mask, vision_mask, is_distill=True)

            # Get 7-class predictions
            logits_7 = out['emotion_logits']  # (B, 7)
            pred_classes = logits_7.argmax(dim=-1).cpu().numpy()  # (B,)

            # Convert continuous labels to 7-class
            true_continuous = L.cpu().numpy()
            true_classes = continuous_to_7class(true_continuous)

            all_preds.extend(pred_classes.tolist())
            all_trues.extend(true_classes.tolist())
            all_true_continuous.extend(true_continuous.tolist())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    # Per-class accuracy
    per_class_acc = {}
    per_class_count = {}
    per_class_correct = {}

    for cls_idx in range(7):
        mask = (all_trues == cls_idx)
        count = mask.sum()
        if count > 0:
            correct = (all_preds[mask] == cls_idx).sum()
            per_class_acc[cls_idx] = float(correct) / count
            per_class_count[cls_idx] = int(count)
            per_class_correct[cls_idx] = int(correct)
        else:
            per_class_acc[cls_idx] = 0.0
            per_class_count[cls_idx] = 0
            per_class_correct[cls_idx] = 0

    # Confusion matrix: C[i][j] = count of true=i predicted=j
    confusion = np.zeros((7, 7), dtype=int)
    for t, p in zip(all_trues, all_preds):
        confusion[int(t)][int(p)] += 1

    # Overall accuracy
    overall_acc = float((all_preds == all_trues).sum()) / len(all_trues)

    return {
        'per_class_acc': {str(k): round(v, 4) for k, v in per_class_acc.items()},
        'per_class_count': {str(k): v for k, v in per_class_count.items()},
        'per_class_correct': {str(k): v for k, v in per_class_correct.items()},
        'confusion_matrix': confusion.tolist(),
        'overall_acc': round(overall_acc, 4),
        'emotion_labels': {str(k): v for k, v in EMOTION_LABELS.items()},
    }


def make_args(text_mode='mp', mode='aligned'):
    """Create args namespace matching the model's expected configuration."""
    from argparse import Namespace
    aligned = (mode == 'aligned')
    return Namespace(
        dataset_name='mosi',
        model_name='mer-mtl',
        featurePath=f"./data/{mode}_50.pkl",
        train_mode='regression',
        KeyEval='Acc_7',
        need_data_aligned=aligned,
        text_mode=text_mode,
        checkpoint_dir='./pt/mermtl',
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
        learning_rate=0.0001,
        grad_clip=0.6,
        patience=5,
        weight_decay=0.0,
        update_epochs=1,
        early_stop=15,
        batch_size=16,
        attn_mask=False,
        use_uncertainty=True,
        initial_log_sigma=0.0,
        aux_task_weight=0.1,
        epochs=30,
        seeds=[42],
        save_name=f'mer_mtl_{mode}_{text_mode}_7cls.pth',
    )


def main():
    parser = argparse.ArgumentParser(description='Extract per-class accuracy for MER-MTL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data/aligned_50.pkl', help='Path to data pkl')
    parser.add_argument('--text_mode', type=str, default='mp', choices=['tt', 'mp'])
    parser.add_argument('--mode', type=str, default='aligned', choices=['aligned', 'unaligned'])
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model_args = make_args(text_mode=args.text_mode, mode=args.mode)
    model = load_model(args.checkpoint, model_args)
    model = model.to(args.device)

    test_data = load_data(args.data, mode=args.mode)

    print(f"Extracting per-class accuracy (device={args.device})...")
    results = extract_per_class(model, test_data, device=args.device)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Per-class Accuracy: {args.mode} / {args.text_mode}")
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

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == '__main__':
    main()
