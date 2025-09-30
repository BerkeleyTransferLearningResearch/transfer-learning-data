# 31-Class Transfer Learning Experiment Framework

This is a complete framework for multi-dataset transfer learning experiments, supporting various initialization modes, automated grid search, and detailed performance evaluation.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Parameter Description](#parameter-description)
- [Experiment Modes](#experiment-modes)
- [Output Description](#output-description)
- [Advanced Features](#advanced-features)
- [Examples](#examples)

## ✨ Features

### Core Features
- **Multiple Initialization Modes**: Support for ImageNet pretraining, random initialization, source domain pretraining, etc.
- **Flexible Data Splitting**: Fixed 10% test set, customizable training set ratio
- **Grid Search Experiments**: Automated batch experiments, supporting multiple configuration combinations
- **Early Stopping Mechanism**: Pretraining phase supports validation set and early stopping to avoid overfitting
- **Mixed Precision Training**: Uses AMP to accelerate training and reduce memory usage
- **Detailed Performance Evaluation**: Including accuracy, recall, F1, AUROC and other metrics
- **Enhanced Classification Report**: Per-class specificity, balanced accuracy, OVR-AUC, etc.

### Data Processing
- Stratified sampling to ensure class balance
- Support for merging multiple datasets
- Automatic class mapping alignment
- Data augmentation (random flip, rotation)

### Training Optimization
- AdamW optimizer + Cosine annealing learning rate
- Automatic mixed precision training
- Asynchronous result saving
- GPU acceleration (CUDA support)

## 🔧 Requirements

### Dependencies
```bash
torch
torchvision
numpy
scikit-learn
pandas
```


## 🚀 Quick Start

### 1. Configure Datasets

First configure dataset paths in `config.py`:

```python
DATASETS_MAP = {
    "dataset1": "/path/to/dataset1",
    "dataset2": "/path/to/dataset2",
    "dataset3": "/path/to/dataset3",
}
```

### 2. Run Basic Experiment

```bash
# Activate conda environment
conda activate mlp

# Run with default configuration
python main.py

# Custom configuration
python main.py \
    --fixed_source dataset1 \
    --splits 0.1 0.3 0.5 \
    --init_modes resnet50_imagenet random \
    --epochs_finetune 10 20 30 \
    --batch_size 32
```

### 3. Generate Experiment Summary

```bash
# Generate CSV summary from JSON files
python main.py --summary --out_dir exp_out
```

## 📖 Parameter Description

### Dataset Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--fixed_source` | str | First dataset | Fixed source dataset for pretraining |
| `--target_mode` | str | `merged` | Target mode: `merged` or `separate` |

### Training Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--splits` | float[] | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` | Training set ratio list (based on training set) |
| `--epochs_finetune` | int[] | `[100]` | Fine-tuning epochs list |
| `--epochs_pretrain` | int | `50` | Pretraining epochs |
| `--batch_size` | int | `32` | Batch size |
| `--lr` | float | `0.001` | Learning rate |
| `--img_size` | int | `224` | Image size |

### Model Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--init_modes` | str[] | `[resnet50_imagenet, random]` | Initialization mode list |
| `--no_amp` | flag | False | Disable mixed precision training |

### Early Stopping Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--val_split` | float | `0.2` | Pretraining validation split ratio |
| `--patience` | int | `5` | Early stopping patience value |

### Other Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--seed` | int | `42` | Random seed |
| `--num_workers` | int | `4` | Number of data loading threads |
| `--out_dir` | str | `exp_out` | Output directory |
| `--summary` | flag | False | Only generate summary file |

## 🎯 Experiment Modes

### 1. resnet50_imagenet
Use ImageNet pretrained ResNet50 model, directly fine-tune on target domain.

**Use Case**: Target domain similar to natural images

```bash
python main.py --init_modes resnet50_imagenet
```

### 2. random
Completely random initialization, train from scratch.

**Use Case**: Comparison baseline, understand pretraining value

```bash
python main.py --init_modes random
```

### 3. random_then_fixed_src
After random initialization, first pretrain on fixed source domain, then fine-tune on target domain.

**Use Case**: Strong correlation between source and target domains

```bash
python main.py \
    --init_modes random_then_fixed_src \
    --fixed_source dataset1 \
    --epochs_pretrain 50
```

## 📊 Output Description

### File Structure

```
exp_out/
├── source_to_target_resnet50_imagenet_r01_e10.json
├── source_to_target_resnet50_imagenet_r01_e20.json
├── source_to_target_random_r03_e10.json
└── summary.csv
```

### JSON File Format

Each experiment generates a JSON file containing:

```json
{
  "target_datasets": ["dataset1", "dataset2"],
  "split": 0.5,
  "init_mode": "resnet50_imagenet",
  "source_dataset": "None",
  "epochs_finetune": 20,
  "acc": 0.8523,
  "recall": 0.8456,
  "precision": 0.8589,
  "f1_macro": 0.8521,
  "f1_micro": 0.8523,
  "auroc": 0.9234,
  "cls_report": "..."
}
```

### Metrics Description

- **acc**: Overall accuracy
- **recall**: Macro-averaged recall
- **precision**: Macro-averaged precision
- **f1_macro**: Macro-averaged F1 score
- **f1_micro**: Micro-averaged F1 score
- **auroc**: Macro-averaged AUROC (OVR)
- **cls_report**: Detailed per-class report (including specificity, balanced accuracy, etc.)

### Summary CSV

Contains summary information of all experiments, sorted by:
1. target_datasets
2. init_mode
3. split
4. epochs_finetune

## 🔬 Advanced Features

### 1. Stratified Sampling

Automatically ensures consistent class proportions in training/test sets:

```python
# Fixed test set 10%, training candidate set 90%
# split parameter controls what proportion of training candidate to use
```

### 2. Balanced Stratified Sampling (Multiple Dataset Merging)

When merging multiple datasets, ensure equal number of samples from each dataset in each class:

```python
# Automatically balance contributions from different datasets
target_mode="merged"
```

### 3. Multi-Epoch Testing

One training session, test at multiple epoch points:

```bash
python main.py --epochs_finetune 10 20 30 40 50
# Will save test results at epochs 10, 20, 30, 40, 50 respectively
```

### 4. Skip Completed Experiments

Automatically detect existing result files and skip duplicate experiments:

```python
# If output file already exists, automatically skip that experiment
```

### 5. Enhanced Classification Report

In addition to traditional metrics, also includes:
- **Specificity**: TNR, True Negative Rate
- **Balanced Accuracy**: (TPR + TNR) / 2
- **Per-class OVR-AUC**: One-vs-Rest AUROC

## 📝 Examples

### Example 1: Complete Grid Search

```bash
python main.py \
    --fixed_source dataset1 \
    --splits 0.1 0.3 0.5 0.7 0.9 \
    --init_modes resnet50_imagenet random random_then_fixed_src \
    --epochs_finetune 10 20 30 40 50 \
    --epochs_pretrain 50 \
    --batch_size 32 \
    --lr 0.001 \
    --target_mode merged
```

### Example 2: Quick Test

```bash
python main.py \
    --splits 0.5 \
    --init_modes resnet50_imagenet \
    --epochs_finetune 10 \
    --batch_size 64
```

### Example 3: Pretraining Experiment

```bash
python main.py \
    --init_modes random_then_fixed_src \
    --fixed_source dataset1 \
    --epochs_pretrain 100 \
    --val_split 0.2 \
    --patience 10 \
    --splits 0.5 0.9
```

### Example 4: Transfer to Multiple Targets Separately

```bash
python main.py \
    --target_mode separate \
    --fixed_source dataset1 \
    --splits 0.5 \
    --init_modes resnet50_imagenet random
# Will transfer to dataset2 and dataset3 separately
```


