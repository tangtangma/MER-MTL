# MER-MTL

Adaptive Multitask Learning with Uncertainty-Aware Loss Balancing for Multimodal Emotion Recognition.

## About

MER-MTL jointly performs sentiment regression and auxiliary emotion classification tasks. It uses learnable uncertainty parameters for automatic loss weighting, with two text encoding modes: Text Transformer (TT) and Mean Pooling (MP).

## Usage

### Single Experiment

```bash
# Run one experiment at a time
python main.py --seed 42 --mode unaligned
python main.py --seed 42 --mode aligned
# ... repeat for seeds 1111, 1112, 1113
```

### Batch Execution

```bash
# Run all 4 seeds for both modes at once
python scripts/run_mer_mtl.py

# Generate paper figures
python scripts/plot_all_figures.py \
    --results_dir ./results \
    --logs_dir ./logs/mermtl \
    --output_dir ./figures
```
