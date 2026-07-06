"""
Per-class Accuracy Extraction for MER-MTL (v2)
==============================================
Uses the project's MMDataset + collate_fn to load data correctly,
handles both token-ID and pre-extracted feature formats.

Usage:
    python extract_per_class_accuracy.py \
        --checkpoint ./pt/mermtl/mer_mtl_aligned_tt_7cls.pth \
        --data ./data/aligned_50.pkl \
        --text_mode tt \
        --mode aligned \
        --output ./results/per_class/tt_aligned.json

Output: JSON file with per-class accuracy and confusion matrix.
"""
import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root so we can import data.dataset and models
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from data.dataset import MMDataset, TokenToEmbedding


# =============================================================================
# MOSI 7-class label mapping
# =============================================================================
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
    """Convert continuous sentiment labels to 7-class indices.
    CMU-MOSI labels are in range [-3, +3]. Standard 7-class binning:
      Class 0: [-3.0, -2.0) Very Negative
      Class 1: [-2.0, -1.0) Negative
      Class 2: [-1.0,  0.0) Somewhat Negative
      Class 3: [ 0.0,  0.0] Neutral  (label == 0)
      Class 4: ( 0.0,  1.0] Somewhat Positive
      Class 5: ( 1.0,  2.0] Positive
      Class 6: ( 2.0,  3.0] Very Positive
    Uses boundaries [-3, -2, -1, 0, 1, 2, 3] with digitize.
    """
    # Boundaries: -3, -2, -1, 0, 1, 2, 3
    # digitize returns index of bin each value falls into
    # values < -3 -> 0, [-3,-2) -> 1, ..., (3,inf) -> 7
    boundaries = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    indices = np.digitize(labels, boundaries)
    # Map: 0 (label<-3) -> 0, 1 ([-3,-2)) -> 0, ..., 7 (label>3) -> 6
    indices = np.clip(indices - 1, 0, 6)
    # Special case: label == 0 should be class 3 (Neutral)
    # digitize(0, boundaries) returns 4 (between 0 and 1), so 4-1=3, correct!
    return indices


# =============================================================================
# Model loading
# =============================================================================
def load_model(checkpoint_path, args):
    """Load MER-MTL model from checkpoint."""
    try:
        from models.mer_mtl_model import MERMTLModel
    except ImportError:
        from mer_mtl_model import MERMTLModel

    model = MERMTLModel(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# =============================================================================
# Load test split using MMDataset (handles token-id format + normalization)
# =============================================================================
def load_test_split(data_path, mode='aligned', text_mode='tt'):
    """
    Load the test split using MMDataset so that:
    - Token IDs are converted to 768-d embeddings via TokenToEmbedding
    - Audio/Vision features are normalized the same way as training
    - Returns numpy arrays ready for inference
    """
    # Create minimal args for MMDataset
    ds_args = Namespace(featurePath=data_path)

    # First pass: detect if token-id format and create shared TokenToEmbedding
    import pickle
    with open(data_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    train_text = raw.get('train', {}).get('text')
    is_token_ids = (train_text is not None and
                    len(train_text.shape) == 3 and train_text.shape[1] == 3)

    token_to_emb = TokenToEmbedding() if is_token_ids else None
    if is_token_ids:
        print("  Detected BERT token IDs format, using TokenToEmbedding")

    # Load test split
    test_ds = MMDataset(ds_args, mode='test', token_to_emb=token_to_emb)

    # Extract all samples
    all_text, all_audio, all_vision, all_labels = [], [], [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        all_text.append(sample['text'].numpy())
        all_audio.append(sample['audio'].numpy())
        all_vision.append(sample['vision'].numpy())
        all_labels.append(sample['labels']['M'].numpy())

    text_arr = np.stack(all_text, axis=0)    # (N, T, 768)
    audio_arr = np.stack(all_audio, axis=0)   # (N, T, 5)
    vision_arr = np.stack(all_vision, axis=0) # (N, T, 20)
    labels_arr = np.concatenate(all_labels, axis=0)  # (N,)

    print(f"  Test set: {text_arr.shape[0]} samples")
    print(f"  Text: {text_arr.shape}, Audio: {audio_arr.shape}, Vision: {vision_arr.shape}")
    print(f"  Labels range: [{labels_arr.min():.2f}, {labels_arr.max():.2f}]")

    return {
        'text': text_arr,
        'audio': audio_arr,
        'vision': vision_arr,
        'labels': labels_arr,
    }


# =============================================================================
# Prepare batch for model input
# =============================================================================
def prepare_batch(text, audio, vision, labels, device='cpu'):
    """Prepare a batch of numpy arrays for model inference."""
    T = torch.tensor(text, dtype=torch.float32).to(device)
    A = torch.tensor(audio, dtype=torch.float32).to(device)
    V = torch.tensor(vision, dtype=torch.float32).to(device)
    L = torch.tensor(labels, dtype=torch.float32).to(device)
    return T, A, V, L


# =============================================================================
# Extract per-class accuracy
# =============================================================================
def extract_per_class(model, test_data, device='cpu', batch_size=32):
    """
    Extract per-class predictions and compute per-class accuracy.
    Returns dict with per_class_acc, confusion_matrix, overall_acc, etc.
    """
    text = test_data['text']
    audio = test_data['audio']
    vision = test_data['vision']
    labels_continuous = test_data['labels']

    N = len(labels_continuous)
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            T, A, V, L = prepare_batch(
                text[start:end], audio[start:end],
                vision[start:end], labels_continuous[start:end],
                device=device
            )
            out = model(T, A, V, None, None, None, is_distill=True)

            # 7-class predictions
            logits_7 = out['emotion_logits']  # (B, 7)
            pred_classes = logits_7.argmax(dim=-1).cpu().numpy()

            # Convert continuous labels to 7-class
            true_classes = continuous_to_7class(labels_continuous[start:end])

            all_preds.extend(pred_classes.tolist())
            all_trues.extend(true_classes.tolist())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    # Per-class accuracy
    per_class_acc = {}
    per_class_count = {}
    per_class_correct = {}

    for cls_idx in range(7):
        mask = (all_trues == cls_idx)
        count = int(mask.sum())
        if count > 0:
            correct = int((all_preds[mask] == cls_idx).sum())
            per_class_acc[cls_idx] = round(correct / count, 4)
            per_class_count[cls_idx] = count
            per_class_correct[cls_idx] = correct
        else:
            per_class_acc[cls_idx] = 0.0
            per_class_count[cls_idx] = 0
            per_class_correct[cls_idx] = 0

    # Confusion matrix: C[i][j] = count of true=i predicted=j
    confusion = np.zeros((7, 7), dtype=int)
    for t, p in zip(all_trues, all_preds):
        confusion[int(t)][int(p)] += 1

    overall_acc = float((all_preds == all_trues).sum()) / len(all_trues)

    return {
        'per_class_acc': {str(k): v for k, v in per_class_acc.items()},
        'per_class_count': {str(k): v for k, v in per_class_count.items()},
        'per_class_correct': {str(k): v for k, v in per_class_correct.items()},
        'confusion_matrix': confusion.tolist(),
        'overall_acc': round(overall_acc, 4),
        'emotion_labels': {str(k): v for k, v in EMOTION_LABELS.items()},
    }


# =============================================================================
# Make model args
# =============================================================================
def make_args(text_mode='mp', mode='aligned'):
    """Create args namespace matching MERMTLModel's expected configuration."""
    return Namespace(
        dataset_name='mosi',
        model_name='mer-mtl',
        featurePath=f"./data/{mode}_50.pkl",
        train_mode='regression',
        KeyEval='Acc_7',
        need_data_aligned=(mode == 'aligned'),
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


# =============================================================================
# Main
# =============================================================================
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

    print(f"Loading test data from: {args.data}")
    test_data = load_test_split(args.data, mode=args.mode, text_mode=args.text_mode)

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
