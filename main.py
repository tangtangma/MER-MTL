"""
MER-MTL Main Entry (Single Run)
============================================================
Usage:
    python main.py --text_mode tt --mode aligned --seed 42
    python main.py --text_mode mp --mode unaligned --seed 1111

Directory structure:
    logs/mermtl/{exp_name}/training.log     - Training logs
    results/mermtl/{exp_name}/final_results.txt - Human-readable results
    results/mermtl/{exp_name}/metrics.json  - Machine-readable results
    checkpoints/mermtl/{save_name}.pth      - Model checkpoints
============================================================
"""
import os
import sys
import gc
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from models.mer_mtl_model import MERMTLModel
from experiments.mer_mtl_trainer import MERMTLTrainer, get_args

# dataset.py is in data/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from dataset import MMDataLoader


def setup_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_dir):
    """Setup logger"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(Path(log_dir) / "training.log")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def print_header(args, seed, cls_mode):
    """Print experiment header"""
    print("=" * 70)
    print("MER-MTL Training (DMD-aligned)")
    print("=" * 70)
    text_mode_desc = "Text Transformer (TT)" if args.text_mode == 'tt' else "Mean Pooling (MP)"
    print(f"Architecture: DMD-aligned (Conv1d proj + {text_mode_desc} + Crossmodal)")
    print(f"Text Mode: {args.text_mode.upper()}")
    print(f"Data Source: {args.featurePath}")
    print(f"Data Mode: {'ALIGNED' if args.need_data_aligned else 'UNALIGNED'}")
    print(f"  - Aligned: all modalities time-synced, masks=None")
    print(f"  - Unaligned: per-modality masks for handling variable lengths")
    print(f"Classification Mode: {cls_mode}")
    print(f"  - 7-class: CrossEntropy (main task)")
    print(f"  - Binary: BCEWithLogitsLoss (auxiliary)")
    print(f"Seed: {seed}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Main Task Weight: 1.0 (high)")
    print(f"Auxiliary Task Weight: {args.aux_task_weight} (uncertainty weighted)")
    print(f"Uncertainty Weighting: {args.use_uncertainty}")
    print("=" * 70)


def run_single_experiment(text_mode='tt', mode='aligned', cls_mode='7cls',
                          seed=42, epochs=30, batch_size=16, lr=0.0001,
                          aux_weight=0.1, data_path=None,
                          log_dir='./logs/mermtl', results_dir='./results/mermtl',
                          checkpoint_dir='./pt/mermtl', use_uncertainty=True):
    """
    Run a single MER-MTL experiment. Can be called directly or via CLI.
    Returns the final test metrics dict, or None on failure.
    """
    # Set seeds
    setup_seed(seed)

    # Build config
    args = get_args(
        mode=mode,
        cls_mode=cls_mode,
        epochs=epochs,
        batch=batch_size,
        lr=lr,
        aux_weight=aux_weight,
        text_mode=text_mode,
        checkpoint_dir=checkpoint_dir,
    )

    args.featurePath = data_path if data_path else f"./data/{mode}_50.pkl"
    args.use_uncertainty = use_uncertainty

    # Build experiment name
    exp_name = f"MER_MTL_{text_mode}_{mode}_seed{seed}"

    # Setup logging
    log_path = os.path.join(log_dir, exp_name)
    logger = setup_logger(log_path)

    # Setup results directory
    res_dir = os.path.join(results_dir, exp_name)
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    print_header(args, seed, cls_mode)

    # Check data file
    if not os.path.exists(args.featurePath):
        print(f"\nERROR: Data file not found: {args.featurePath}")
        print("Please place MOSI data files at:")
        print(f"  ./data/aligned_50.pkl")
        print(f"  ./data/unaligned_50.pkl")
        return None

    # Load data
    logger.info("Loading dataset...")
    dataloader = MMDataLoader(args, num_workers=0)
    train_size = len(dataloader['train'].dataset)
    valid_size = len(dataloader['valid'].dataset)
    test_size = len(dataloader['test'].dataset)
    logger.info(f"Dataset: {train_size} train / {valid_size} valid / {test_size} test")

    # Create model
    logger.info("Creating model: MER-MTL (DMD-aligned)...")
    model = MERMTLModel(args)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    model = model.to(device)

    # Create trainer
    trainer = MERMTLTrainer(args, device)

    # Train
    logger.info("\n" + "=" * 70)
    logger.info("Training started")
    logger.info("=" * 70)

    train_results = trainer.do_train(model, dataloader, return_epoch_results=True)

    # Load best model and test
    logger.info("\nLoading best model for final test...")
    ckpt_path = Path(checkpoint_dir) / args.save_name
    model.load_state_dict(torch.load(ckpt_path))
    final_test = trainer.do_test(model, dataloader['test'], mode="TEST")

    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info(f"TextMode: {text_mode}, Mode: {mode}, Cls: {cls_mode}, Seed: {seed}")
    logger.info(f"Best Test Acc-7: {final_test['Acc_7']:.2%}")
    logger.info(f"Best Test Acc-2: {final_test['Acc_2']:.2%}")
    logger.info(f"Best Test F1: {final_test['F1_score']:.4f}")
    logger.info(f"Best Test MAE: {final_test['MAE']:.4f}")
    logger.info(f"Best Test Corr: {final_test['Corr']:.4f}")
    logger.info("=" * 70)

    # Save results
    txt_path = Path(res_dir) / "final_results.txt"
    with open(txt_path, 'w') as f:
        f.write(f"MER-MTL (DMD-aligned) Results\n")
        f.write(f"TextMode: {text_mode}\n")
        f.write(f"Mode: {mode} ({'aligned' if args.need_data_aligned else 'unaligned'})\n")
        f.write(f"Cls: {cls_mode}, Seed: {seed}\n")
        f.write(f"Best Test Acc-7: {final_test['Acc_7']:.2%}\n")
        f.write(f"Best Test Acc-2: {final_test['Acc_2']:.2%}\n")
        f.write(f"Best Test F1: {final_test['F1_score']:.4f}\n")
        f.write(f"Best Test MAE: {final_test['MAE']:.4f}\n")
        f.write(f"Best Test Corr: {final_test['Corr']:.4f}\n")
    logger.info(f"Results saved to: {txt_path}")

    json_path = Path(res_dir) / "metrics.json"
    metrics_data = {
        'model': 'MER-MTL',
        'text_mode': text_mode,
        'mode': mode,
        'cls_mode': cls_mode,
        'seed': seed,
        'acc7': final_test['Acc_7'],
        'acc2': final_test['Acc_2'],
        'f1': final_test['F1_score'],
        'mae': final_test['MAE'],
        'corr': final_test['Corr'],
        'f1_binary': final_test.get('F1_binary', 0.0),
        'f1_weighted': final_test.get('F1_weighted', 0.0),
        'mae_class': final_test.get('MAE_class', 0.0),
        'acc2_from7': final_test.get('Acc_2_from7', 0.0),
        'acc2_direct': final_test.get('Acc_2_direct', 0.0),
    }
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Metrics saved to: {json_path}")

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return final_test


def main():
    parser = argparse.ArgumentParser(description='MER-MTL Single Run')
    parser.add_argument('--model', type=str, default='mer_mtl',
                       help='Model name (default: mer_mtl, reserved for compatibility)')
    parser.add_argument('--mode', type=str, default='aligned',
                       choices=['aligned', 'unaligned'],
                       help='Data source: aligned or unaligned')
    parser.add_argument('--cls_mode', type=str, default='7cls',
                       choices=['7cls', 'binary', 'both'],
                       help='Classification mode: 7cls, binary, or both')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--aux_weight', type=float, default=0.1,
                       help='Auxiliary task weight (scaling factor)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Data file path (default: ./data/{mode}_50.pkl)')
    parser.add_argument('--log_dir', type=str, default='./logs/mermtl',
                       help='Log directory (default: ./logs/mermtl)')
    parser.add_argument('--results_dir', type=str, default='./results/mermtl',
                       help='Results directory (default: ./results/mermtl)')
    parser.add_argument('--checkpoint_dir', type=str, default='./pt/mermtl',
                       help='Checkpoint directory (default: ./pt/mermtl)')
    parser.add_argument('--text_mode', type=str, default='tt',
                       choices=['tt', 'mp'],
                       help='Text processing mode: tt (Text Transformer) or mp (Mean Pooling)')
    parser.add_argument('--use_uncertainty', action='store_true', default=True,
                       help='Use uncertainty weighting on aux tasks')
    parser.add_argument('--no_uncertainty', action='store_true',
                       help='Disable uncertainty weighting')
    args_cmd = parser.parse_args()

    run_single_experiment(
        text_mode=args_cmd.text_mode,
        mode=args_cmd.mode,
        cls_mode=args_cmd.cls_mode,
        seed=args_cmd.seed,
        epochs=args_cmd.epochs,
        batch_size=args_cmd.batch_size,
        lr=args_cmd.lr,
        aux_weight=args_cmd.aux_weight,
        data_path=args_cmd.data_path,
        log_dir=args_cmd.log_dir,
        results_dir=args_cmd.results_dir,
        checkpoint_dir=args_cmd.checkpoint_dir,
        use_uncertainty=not args_cmd.no_uncertainty,
    )


if __name__ == "__main__":
    main()
