# MER-MTL
## Multi-modal Emotion Recognition with Multi-Task Learning

Unified codebase for MER-MTL supporting DMD baseline and MER-MTL (with TT and MP text encoding variants).

---

## Project Structure

```
MER-MTL/
├── config.py              # Configuration (DMD/MER-MTL)
├── main.py                # Unified entry point (single experiment)
├── requirements.txt       # Dependencies
├── figures/               # Visualization output directory
├── results/               # Experiment results output directory
│   ├── DMD/
│   │   ├── aligned_seed42/
│   │   ├── aligned_seed1111/
│   │   ├── aligned_seed1112/
│   │   ├── aligned_seed1113/
│   │   ├── unaligned_seed42/
│   │   ├── unaligned_seed1111/
│   │   ├── unaligned_seed1112/
│   │   └── unaligned_seed1113/
│   ├── MER_MTL_TT/       # Transformer Text encoder
│   └── MER_MTL_MP/       # MeanPooling text encoder
├── data/                  # Data loading
├── models/                # Model definitions
│   ├── dmd_model.py       # DMD baseline
│   ├── mer_mtl_model.py   # MER-MTL unified model
│   └── ...
├── experiments/           # Trainers
│   ├── dmd_trainer.py
│   ├── mer_mtl_trainer.py
│   └── base_trainer.py
├── utils/                 # Utilities
│   ├── logger.py
│   ├── metrics.py
│   └── visualization.py   # Visualization (includes comparison plot generation)
└── scripts/
    ├── run_mer_mtl.py     # Batch run script
    └── visualize_comparison.py  # Comprehensive visualization script
```

---

## Model Description

| Model | Description | Auxiliary Tasks | Weight Strategy |
|-------|-------------|-----------------|-----------------|
| **DMD** | Original DMD baseline | 4 (L_rec, L_cyc, L_mar, L_ort) | Manual λ weights |
| **MER-MTL-TT** | MER-MTL + Transformer text encoder | 4 + uncertainty weighting | Kendall 2018 |
| **MER-MTL-MP** | MER-MTL + MeanPooling text encoder | 4 + uncertainty weighting | Kendall 2018 |

---

## Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- seaborn

### Data Preparation

Ensure data files exist:
- Aligned mode: `./data/aligned_50.pkl`
- Unaligned mode: `./data/unaligned_50.pkl`

Or specify custom path via `--data_path` parameter.

### Single Experiment

```bash
# DMD baseline
python main.py --model dmd --mode aligned --seed 42
python main.py --model dmd --mode unaligned --seed 42

# MER-MTL TT (Transformer Text)
python main.py --model mer_mtl --use_text_transformer True --mode aligned --seed 42
python main.py --model mer_mtl --use_text_transformer True --mode unaligned --seed 42

# MER-MTL MP (MeanPooling)
python main.py --model mer_mtl --use_text_transformer False --mode aligned --seed 42
python main.py --model mer_mtl --use_text_transformer False --mode unaligned --seed 42
```

### Batch Run

```bash
# Run all MER-MTL experiments (2 models × 2 modes × 4 seeds = 16 groups)
python scripts/run_mer_mtl.py all

# Run TT model only
python scripts/run_mer_mtl.py all --model_type tt

# Run MP model only
python scripts/run_mer_mtl.py all --model_type mp

# Run all DMD experiments
python scripts/run_mer_mtl.py dmd
```

### Generate Comparison Visualizations

```bash
# Generate demo visualization with sample data
python scripts/visualize_comparison.py --demo

# Generate comparison from experiment results
python scripts/visualize_comparison.py --results_dir ./results --output ./figures
```

---

## Experiment Matrix

| Model | Mode | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|-------|------|--------|--------|--------|--------|
| DMD | Aligned | ✓ | ✓ | ✓ | ✓ |
| DMD | Unaligned | ✓ | ✓ | ✓ | ✓ |
| MER-MTL-TT | Aligned | ✓ | ✓ | ✓ | ✓ |
| MER-MTL-TT | Unaligned | ✓ | ✓ | ✓ | ✓ |
| MER-MTL-MP | Aligned | ✓ | ✓ | ✓ | ✓ |
| MER-MTL-MP | Unaligned | ✓ | ✓ | ✓ | ✓ |

**Total: 6 configurations × 4 seeds = 24 experiment groups**

---

## Auxiliary Tasks Description

MER-MTL shares the same 4 auxiliary tasks with DMD:

| Task | Description | Loss Function |
|------|-------------|---------------|
| L_rec | Modality reconstruction | MSE |
| L_cyc | Cyclic consistency | MSE |
| L_mar | Modality alignment relation | BCE |
| L_ort | Orthogonality regularization | Frobenius norm |

**Uncertainty Weighting**: MER-MTL uses Kendall 2018's method to automatically learn each task weight σ_k, no manual tuning required.

---

## Parameter Description

### main.py Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model type: `dmd` or `mer_mtl` | Required |
| `--use_text_transformer` | TT=True or MP=False | True |
| `--mode` | `aligned` or `unaligned` | aligned |
| `--seed` | Random seed: 42, 1111, 1112, 1113 | 42 |
| `--epochs` | Training epochs | 30 |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--data_path` | Data path | Auto-select |

### visualize_comparison.py Parameters

| Parameter | Description |
|-----------|-------------|
| `--results_dir` | Results directory (default: `./results`) |
| `--csv` | Load results from CSV |
| `--output` | Output directory (default: `./figures`) |
| `--demo` | Generate demo with sample data |

---

## Output Description

### Training Output

- **Logs**: `./logs/{model_name}_{mode}_seed{seed}/`
- **Results**: `./results/{model_name}/{mode}_seed{seed}/metrics.json`

### metrics.json Format

```json
{
  "acc7": 0.412,
  "acc2": 0.831,
  "f1": 0.825,
  "mae": 0.823,
  "corr": 0.456
}
```

### Visualization Output

- `figures/model_comparison/comparison_acc7_grouped.png` - Acc-7 grouped bar chart
- `figures/model_comparison/comparison_acc7_heatmap.png` - Acc-7 heatmap
- `figures/model_comparison/comparison_all_metrics.png` - Multi-metric comparison
- `figures/training_curves/` - Training curves
- `figures/confusion_matrix/` - Confusion matrices
- `figures/tsne/` - t-SNE visualizations

---

## Data Dimensions

| Modality | Dimension | Description |
|----------|-----------|-------------|
| Text | 768 | BERT embedding features |
| Audio | 5 | Audio features |
| Vision | 20 | Visual features |
| Sequence | 50 | Temporal steps |

---

## Evaluation Metrics

- **Acc-7**: 7-class sentiment classification accuracy (-3 to +3)
- **Acc-2**: Binary sentiment classification accuracy (positive/negative)
- **F1**: Binary F1 score
- **MAE**: Mean Absolute Error
- **Corr**: Pearson correlation coefficient

---

## Contact

For questions or suggestions, please contact the project maintainer.
