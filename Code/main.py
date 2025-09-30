# transfer_experiments.py
# -*- coding: utf-8 -*-
import os
import time
import json
import math
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report as sk_classification_report,
    roc_auc_score,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.multiclass import unique_labels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models

import warnings
warnings.filterwarnings("ignore")

# Import configuration file
from config import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    ImageConfig,
    OutputConfig,
    CLIDefaults,
    get_default_config,
    validate_config,
)

# Thread lock for file writing
file_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO, format=OutputConfig.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Keep these constants for backward compatibility
IMAGENET_MEAN = ImageConfig.IMAGENET_MEAN
IMAGENET_STD = ImageConfig.IMAGENET_STD

# ============== Basic Configuration ==============
def seed_all(seed: int = 42):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN more reproducible (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Build training, augmented training and test data transforms"""
    
    # Augmented training transforms - more aggressive data augmentation
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf


def enforce_class_mapping(root: str, classes: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, int]]:
    """
    Return an ImageFolder and ensure class_to_idx order matches classes (if provided).
    If classes not provided, use current root's class names (sorted) as reference.
    """
    # First use ImageFolder to check class names
    tmp = datasets.ImageFolder(root)
    cur_classes = sorted(tmp.class_to_idx.keys())
    if classes is None:
        classes = cur_classes
    else:
        # Validate class sets are consistent
        if set(cur_classes) != set(classes):
            raise ValueError(f"Dataset {root} classes don't match reference:\n"
                             f"Reference: {sorted(classes)}\nCurrent: {sorted(cur_classes)}")



    # Force use reference classes order to construct mapping
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def load_imagefolder_with_mapping(root: str, transform, classes: List[str], class_to_idx: Dict[str, int]) -> datasets.ImageFolder:
    """Load ImageFolder and remap classes"""
    ds = datasets.ImageFolder(root=root, transform=transform)
    # Remap targets to unified class_to_idx
    # Note: ImageFolder internally uses samples (list of (path, target)), rewrite their targets here
    new_samples = []
    for path, old_y in ds.samples:
        cls_name = ds.classes[old_y]
        new_y = class_to_idx[cls_name]
        new_samples.append((path, new_y))
    ds.samples = new_samples
    ds.targets = [y for _, y in new_samples]
    ds.classes = classes
    ds.class_to_idx = class_to_idx
    return ds


def stratified_split_indices(targets: List[int], ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Stratified sampling on ImageFolder targets, return (train_idx, test_idx)
    ratio represents training (fine-tuning) proportion. If ratio==0, training set is empty.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    train_idx, test_idx = [], []
    classes = np.unique(targets)
    for c in classes:
        idxs = np.where(targets == c)[0]
        rng.shuffle(idxs)
        k = int(round(len(idxs) * ratio))
        # Allow k == 0 (i.e., this class doesn't participate in fine-tuning), maintain strict "rest as test" semantics
        train_idx.extend(idxs[:k].tolist())
        test_idx.extend(idxs[k:].tolist())
    # Only shuffle training set, keep test set order to ensure result consistency
    rng.shuffle(train_idx)
    return train_idx, test_idx


def balanced_stratified_split_indices(targets: List[int], dataset_indices: List[int], ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Balanced stratified sampling: ensure equal number of samples from different datasets in each class
    targets: sample class labels
    dataset_indices: dataset index corresponding to samples (0 for first dataset, 1 for second dataset)
    ratio: training proportion
    Returns: (train_idx, test_idx)
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    dataset_indices = np.array(dataset_indices)
    
    train_idx, test_idx = [], []
    classes = np.unique(targets)
    
    for c in classes:
        # Find all samples of this class
        class_mask = targets == c
        class_indices = np.where(class_mask)[0]
        
        # Group by dataset
        dataset1_indices = class_indices[dataset_indices[class_indices] == 0]  # First dataset
        dataset2_indices = class_indices[dataset_indices[class_indices] == 1]  # Second dataset
        
        # Calculate number of training samples to select from each dataset
        n_train_d1 = int(round(len(dataset1_indices) * ratio))
        n_train_d2 = int(round(len(dataset2_indices) * ratio))
        
        # Random selection
        rng.shuffle(dataset1_indices)
        rng.shuffle(dataset2_indices)
        
        train_idx.extend(dataset1_indices[:n_train_d1].tolist())
        train_idx.extend(dataset2_indices[:n_train_d2].tolist())
        test_idx.extend(dataset1_indices[n_train_d1:].tolist())
        test_idx.extend(dataset2_indices[n_train_d2:].tolist())
    
    # Only shuffle training set, keep test set order to ensure result consistency
    rng.shuffle(train_idx)
    
    return train_idx, test_idx



# ============== Model Related ==============
def build_model(num_classes: int,
                init_mode: str = "resnet50_imagenet",
                device: str = "cuda") -> nn.Module:
    """
    Build model
    init_mode:
      - "resnet50_imagenet"      : Full resnet50 (ImageNet pretrained)
      - "random"                 : Completely random weights
      - "random_then_fixed_src"  : Completely random -> supervised pretraining on fixed source data
    """
    if init_mode == "resnet50_imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif init_mode in ["random", "random_then_fixed_src"]:
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model.to(device)


class MetricsCalculator:
    """Metrics calculator"""
    def __init__(self):
        self.cache = {}
    
    def calculate_metrics(self, all_true: List[int], all_pred: List[int], cache_key: str = None) -> Dict:
        """Calculate all metrics"""
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        acc = accuracy_score(all_true, all_pred)
        recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
        precision = precision_score(all_true, all_pred, average="macro", zero_division=0)
        f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
        f1_micro = f1_score(all_true, all_pred, average="micro", zero_division=0)

        out = {
            "acc": float(acc),
            "recall": float(recall),
            "precision": float(precision),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
        }
        
        if cache_key:
            self.cache[cache_key] = out
        
        return out

def classification_report_plus(
    y_true,
    y_pred,
    *,
    y_score=None,                 # New: for computing per-class ovr-AUC
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=False,
    zero_division="warn",
    per_class_accuracy="balanced",  # "balanced" | "recall" | "specificity"
):
    # Parse labels and names
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    if target_names is None:
        target_names = [str(l) for l in labels]

    # Original three metrics
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    # Per-class confusion matrix (one-vs-rest)
    MCM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = MCM[:, 0, 0], MCM[:, 0, 1], MCM[:, 1, 0], MCM[:, 1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)   # TNR
        recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)        # TPR (= r)

    if per_class_accuracy == "balanced":
        per_acc = (recall + specificity) / 2.0
    elif per_class_accuracy == "recall":
        per_acc = recall
    elif per_class_accuracy == "specificity":
        per_acc = specificity
    else:
        raise ValueError("per_class_accuracy must be one of: 'balanced', 'recall', 'specificity'")

    # Per-class ovr-AUC
    aucs = np.full(len(labels), np.nan, dtype=float)
    if y_score is not None:
        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        if y_score_arr.ndim == 1:  # Binary classification y_score is positive class score
            # Find "positive class" label (default greater label)
            labels_arr = np.array(labels)
            try:
                pos_label = max(np.unique(y_true_arr))
                bin_true = (y_true_arr == pos_label).astype(int)
                auc_val = roc_auc_score(bin_true, y_score_arr)
                if labels_arr.size == 2 and pos_label in labels_arr:
                    pos_idx = int(np.where(labels_arr == pos_label)[0][0])
                    neg_idx = 1 - pos_idx
                    aucs[pos_idx] = auc_val
                    aucs[neg_idx] = 1.0 - auc_val
                else:
                    aucs[:] = auc_val
            except Exception:
                pass
        else:
            # Multi-class: one-vs-rest per class
            for i, lab in enumerate(labels):
                bin_true = (y_true_arr == lab).astype(int)
                try:
                    aucs[i] = roc_auc_score(bin_true, y_score_arr[:, i])
                except Exception:
                    aucs[i] = np.nan
    # Build output
    headers = ["precision", "recall", "f1-score", "support", "specificity", "balanced-acc", "ovr-auc"]
    rows = zip(target_names, p, r, f1, s, specificity, per_acc, aucs)

    if output_dict:
        report = {}
        for row in rows:
            name, *vals = row
            report[name] = dict(zip(headers, [float(v) if v is not None and not np.isnan(v) else np.nan for v in vals]))
        # Append macro/weighted/micro averages (keep consistent with original report; new columns filled with NaN)
        base = sk_classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=True,
            zero_division=zero_division,
        )
        for k, v in base.items():
            if isinstance(v, dict):
                report[k] = {
                    "precision": float(v["precision"]),
                    "recall": float(v["recall"]),
                    "f1-score": float(v["f1-score"]),
                    "support": int(v["support"]) if "support" in v and v["support"] is not None else np.sum(s),
                    "specificity": np.nan,
                    "balanced-acc": np.nan,
                    "ovr-auc": np.nan,
                }
            elif k == "accuracy":
                report[k] = float(v)
        return report
    else:
        # Text format
        name_width = max(len(n) for n in target_names + ["weighted avg"])
        width = max(name_width, len("weighted avg"), digits)
        head_fmt = "{:>{width}s} " + " {:>11}" * len(headers)
        row_fmt = "{:>{width}s} " + " {:>11.{digits}f}" * 3 + " {:>11}" + " {:>11.{digits}f}" * 3 + "\n"

        out = head_fmt.format("", *headers, width=width) + "\n\n"
        for row in rows:
            name, prec, rec, f1v, sup, spec, balc, aucv = row
            out += row_fmt.format(
                name, prec, rec, f1v, int(sup), spec, balc, aucv,
                width=width, digits=digits
            )
        out += "\n"

        # Append accuracy / macro / weighted summaries (consistent with original output)
        base_txt = sk_classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=False,
            zero_division=zero_division,
        )
        out += base_txt.split("\n", 1)[1]  # Reuse summary section from original table
        return out

def train_one_stage(model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader] = None,
                    epochs: int = 10,
                    lr: float = 1e-3,
                    device: str = "cuda",
                    use_amp: bool = True,
                    verbose: bool = True,
                    patience: int = 5,
                    min_delta: float = 1e-4,
                    test_loader: Optional[DataLoader] = None,
                    epochs_finetune_list: Optional[List[int]] = None,
                    out_dir: Optional[str] = None,
                    exp_info_base: Optional[Dict] = None) -> Dict:
    """
    Train one stage, supports validation set and early stopping
    Optimization: reduce redundant computation, use more efficient data processing
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Early stopping related variables
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Test related variables
    test_results = []
    if test_loader is not None and epochs_finetune_list is not None:
        logger.info(f"Will test at following epochs: {epochs_finetune_list}")

    # Pre-allocate tensors to reduce memory allocation
    device_obj = torch.device(device)
    
    for ep in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss, total_correct, total_cnt = 0.0, 0, 0
        
        for x, y in train_loader:
            x = x.to(device_obj, non_blocking=True)
            y = y.to(device_obj, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Use more efficient computation
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == y).sum().item()
            total_cnt += batch_size

        train_loss = total_loss / max(1, total_cnt)
        train_acc = total_correct / max(1, total_cnt)
        
        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            model.eval()
            val_total_loss, val_total_correct, val_total_cnt = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device_obj, non_blocking=True)
                    y = y.to(device_obj, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits, y)
                    batch_size = x.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_correct += (logits.argmax(1) == y).sum().item()
                    val_total_cnt += batch_size
            
            val_loss = val_total_loss / max(1, val_total_cnt)
            val_acc = val_total_correct / max(1, val_total_cnt)
            
            # Early stopping check
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Epoch {ep}/{epochs} | Early stopping triggered! Val acc hasn't improved for {patience} consecutive epochs")
                # Restore best model state
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        # Test at specified epochs
        if test_loader is not None and epochs_finetune_list is not None and ep in epochs_finetune_list:
            logger.info(f"Epoch {ep}/{epochs} | Starting testing...")
            test_metrics = evaluate(model, test_loader, device=device, return_report=True)
            
            # Asynchronously save test results
            if out_dir is not None and exp_info_base is not None:
                save_test_result_async(exp_info_base, test_metrics, ep, out_dir, test_results)

        scheduler.step()
        
        if verbose:
            if val_loader is not None:
                logger.info(f"Epoch {ep}/{epochs} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")
            else:
                logger.info(f"Epoch {ep}/{epochs} | Loss {train_loss:.4f} | Acc {train_acc:.4f}")

    return {
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "final_val_loss": val_loss if val_loader is not None else None,
        "final_val_acc": val_acc if val_loader is not None else None,
        "best_val_acc": best_val_acc if val_loader is not None else None,
        "epochs_trained": ep,
        "test_results": test_results
    }


def save_test_result_async(exp_info_base: Dict, test_metrics: Dict, ep: int, out_dir: str, test_results: List):
    """Asynchronously save test results"""
    def save_worker():
        try:
            test_exp_info = exp_info_base.copy()
            test_exp_info.update({
                "epochs_finetune": ep,
                **test_metrics
            })
            
            # Filter out unnecessary training config parameters
            filtered_info = {
                "target_datasets": test_exp_info.get("target_datasets"),
                "split": test_exp_info.get("split"),
                "init_mode": test_exp_info.get("init_mode"),
                "source_dataset": test_exp_info.get("source_dataset"),
                "epochs_finetune": test_exp_info.get("epochs_finetune"),
                "acc": test_exp_info.get("acc"),
                "recall": test_exp_info.get("recall"),
                "precision": test_exp_info.get("precision"),
                "f1_macro": test_exp_info.get("f1_macro"),
                "f1_micro": test_exp_info.get("f1_micro"),
                "auroc": test_exp_info.get("auroc"),
                "cls_report": test_exp_info.get("cls_report")
            }
            
            # Generate filename
            target_datasets = exp_info_base.get("target_datasets", ["unknown"])
            if isinstance(target_datasets, list) and len(target_datasets) > 1:
                target_name_for_file = "_".join(target_datasets)
            else:
                target_name_for_file = target_datasets[0] if isinstance(target_datasets, list) else target_datasets
            
            init_mode = exp_info_base.get("init_mode", "unknown")
            split = exp_info_base.get("split", 0.0)
            
            out_json = Path(out_dir) / f"{exp_info_base['source_dataset']}_to_{target_name_for_file}_{init_mode}_r{str(split).replace('.', '')}_e{ep}.json"
            
            with file_lock:
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(filtered_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Epoch {ep} test results saved: {out_json}")
            
            # Record test results
            test_results.append({
                "epoch": ep,
                "metrics": test_metrics,
                "file_path": str(out_json)
            })
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
    
    # Execute asynchronously using thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(save_worker)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cuda",
             return_report: bool = True) -> Dict:
    """Evaluate model performance, optimize memory usage"""
    model.eval()
    all_pred, all_true = [], []
    all_scores = []
    device_obj = torch.device(device)
    
    for x, y in loader:
        x = x.to(device_obj, non_blocking=True)
        y = y.to(device_obj, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

    # Use metrics calculator
    metrics_calc = MetricsCalculator()
    out = metrics_calc.calculate_metrics(all_true, all_pred)
    
    # Calculate multi-class AUROC (macro, OVR), fallback to None when classes incomplete or scores abnormal
    try:
        auroc = roc_auc_score(np.array(all_true), np.array(all_scores), multi_class="ovr", average="macro")
        out["auroc"] = float(auroc)
    except Exception:
        out["auroc"] = None
    
    if return_report:
        out["cls_report"] = classification_report_plus(all_true, all_pred, y_score=np.array(all_scores), digits=4, zero_division=0)
    
    return out


# ============== Experiment Pipeline ==============
def pretrain_on_fixed_source(model: nn.Module,
                             source_root: str,
                             config: ExperimentConfig,
                             classes: List[str],
                             class_to_idx: Dict[str, int]) -> nn.Module:
    """
    For init_mode == random_then_fixed_src supervised pretraining phase.
    Supports validation split and early stopping mechanism.
    """
    train_tf, test_tf = build_transforms(config.img_size)
    
    # Load source dataset
    src_ds = load_imagefolder_with_mapping(source_root, train_tf, classes, class_to_idx)
    
    # Split training and validation sets
    total_samples = len(src_ds)
    val_size = int(total_samples * config.val_split)
    train_size = total_samples - val_size
    
    # Use stratified sampling to ensure class balance
    targets = src_ds.targets
    train_idx, val_idx = stratified_split_indices(targets, 1 - config.val_split, seed=config.seed)
    
    # Create training and validation subsets
    train_subset = Subset(src_ds, train_idx)
    val_subset = Subset(src_ds, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=(config.use_amp))
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=(config.use_amp))
    
    logger.info(f"Supervised pretraining on fixed source dataset ({source_root})")
    logger.info(f"Training set: {len(train_subset)} samples, validation set: {len(val_subset)} samples")
    logger.info(f"Total samples: {total_samples}, validation split: {config.val_split:.1%}")
    
    # Training with validation set
    train_results = train_one_stage(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        epochs=config.epochs_pretrain, 
        lr=config.lr, 
        device="cuda" if torch.cuda.is_available() else "cpu", 
        use_amp=config.use_amp,
        patience=config.patience
    )
    
    logger.info(f"Pretraining completed! Best validation accuracy: {train_results['best_val_acc']:.4f}")
    logger.info(f"Actual training epochs: {train_results['epochs_trained']}/{config.epochs_pretrain}")
    
    return model


# Continue optimizing other functions...

def build_result_filename(init_mode: str,
                          fixed_source_name: str,
                          merged_target_roots: Optional[List[str]],
                          target_name: str,
                          split: float,
                          epoch: int) -> str:
    """Construct result filename (consistent with saving logic)"""
    source_part = fixed_source_name if init_mode == "random_then_fixed_src" else "None"
    if merged_target_roots and len(merged_target_roots) > 1:
        target_part = "_".join(merged_target_roots)
    else:
        target_part = target_name
    split_part = str(split).replace('.', '')
    return f"{source_part}_to_{target_part}_{init_mode}_r{split_part}_e{epoch}.json"


def run_single_experiment(
        datasets_map: Dict[str, str],
        target_name: str,
        split: float,
        init_mode: str,
        fixed_source_name: str,
        out_dir: str = '/home/exp/exp/ruih/newdr/exp_out',
        epochs_finetune: int = 10,
        epochs_finetune_list: Optional[List[int]] = None,
        epochs_pretrain: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
        seed: int = 42,
        use_amp: bool = True,
        merged_target_roots: Optional[List[str]] = None,
) -> Dict:
    """
    Complete experiment for one target + split + init_mode:
    Optimization: use config class, reduce redundant code, improve readability
    """
    logger.debug(f"Starting experiment: target={target_name}, split={split}, init_mode={init_mode}")
    logger.info(f"Dataset mapping: {list(datasets_map.keys())}")
    if merged_target_roots:
        logger.info(f"Merged target datasets: {merged_target_roots}")
    
    # Create config object
    config = ExperimentConfig(
        batch_size=batch_size,
        lr=lr,
        img_size=img_size,
        num_workers=num_workers,
        seed=seed,
        use_amp=use_amp,
        epochs_pretrain=epochs_pretrain
    )
    
    # Handle epochs_finetune parameter
    if epochs_finetune_list is not None:
        max_epochs = max(epochs_finetune_list)
        logger.info(f"Using multiple fine-tuning epochs: {epochs_finetune_list}, max training epochs: {max_epochs}")
    else:
        max_epochs = epochs_finetune
        epochs_finetune_list = [epochs_finetune]
        logger.info(f"Using single fine-tuning epochs: {epochs_finetune}")
    
    seed_all(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Unified class mapping ===
    names = list(datasets_map.keys())
    first_root = datasets_map[names[0]]
    classes, class_to_idx = enforce_class_mapping(first_root, classes=None)
    # Validate other two
    for k, root in datasets_map.items():
        enforce_class_mapping(root, classes=classes)

    # === Dataset preparation ===
    train_tf, test_tf = build_transforms(config.img_size)

    # Target dataset: if merged datasets, merge multiple datasets
    if merged_target_roots and len(merged_target_roots) > 1:
        # Merge multiple datasets - use train_tf to ensure consistency
        tgt_datasets = []
        for dataset_name in merged_target_roots:
            root = datasets_map[dataset_name]
            ds = load_imagefolder_with_mapping(root, train_tf, classes, class_to_idx)
            tgt_datasets.append(ds)
        
        # Merge datasets
        tgt_full_ds = ConcatDataset(tgt_datasets)
        
        # Recalculate targets list
        all_targets = []
        for ds in tgt_datasets:
            if hasattr(ds, 'targets'):
                all_targets.extend(ds.targets)
            else:
                all_targets.extend([y for _, y in ds.samples])
        
        logger.info(f"Merged target datasets: {merged_target_roots}, total samples: {len(tgt_full_ds)}")
    else:
        # Single dataset case
        tgt_root = datasets_map[target_name]
        tgt_full_ds = load_imagefolder_with_mapping(tgt_root, test_tf, classes, class_to_idx)
        all_targets = tgt_full_ds.targets

    # Fixed test set at 10%, training candidate set at 90%, split decides what proportion of training candidate to use
    # First determine fixed test set indices (10%)
    _, test_idx = stratified_split_indices(all_targets, 1 - 0.1, config.seed)
    # Training candidate set: samples after removing test set (fixed at 90%)
    all_indices = list(range(len(all_targets)))
    test_idx_set = set(test_idx)
    remain_indices = [i for i in all_indices if i not in test_idx_set]
    # split directly represents what proportion of training candidate to use (split proportion of 90%)
    remain_targets = [all_targets[i] for i in remain_indices]
    train_rel_idx, _ = stratified_split_indices(remain_targets, split, config.seed)
    train_idx = [remain_indices[i] for i in train_rel_idx]
    
    train_ratio_of_total = len(train_idx) / len(all_targets) if len(all_targets) > 0 else 0
    logger.info(f"Fixed test ratio 10% | Training candidate 90% | Using {split*100:.0f}% of training candidate | Training samples {len(train_idx)} ({train_ratio_of_total*100:.1f}% of total) | Test samples {len(test_idx)}")
              
    # Create training and test subsets
    if merged_target_roots and len(merged_target_roots) > 1:
        # For merged datasets, need to rebuild training and test data
        train_datasets = []
        test_datasets = []
        
        # Reload each dataset and apply corresponding transform
        for dataset_name in merged_target_roots:
            root = datasets_map[dataset_name]
            train_ds = load_imagefolder_with_mapping(root, train_tf, classes, class_to_idx)
            test_ds = load_imagefolder_with_mapping(root, test_tf, classes, class_to_idx)
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        
        # Merge training and test datasets
        tgt_train_full = ConcatDataset(train_datasets)
        tgt_test_full = ConcatDataset(test_datasets)
        
        # Recalculate targets and dataset_indices (both groups should be in same order)
        train_targets, train_dataset_indices = [], []
        test_targets, test_dataset_indices = [], []
        for i, ds in enumerate(train_datasets):
            if hasattr(ds, 'targets'):
                train_targets.extend(ds.targets)
            else:
                train_targets.extend([y for _, y in ds.samples])
            train_dataset_indices.extend([i] * len(ds))
        for i, ds in enumerate(test_datasets):
            if hasattr(ds, 'targets'):
                test_targets.extend(ds.targets)
            else:
                test_targets.extend([y for _, y in ds.samples])
            test_dataset_indices.extend([i] * len(ds))
        
        # First fix test set at 10% (take 10% from each dataset and each class separately), keep consistent across splits
        fixed_test_ratio = 0.1
        _, test_idx_fixed = balanced_stratified_split_indices(
            test_targets, test_dataset_indices, 1 - fixed_test_ratio, config.seed
        )
        
        # Training candidate set: remove test index positions from training merged set (fixed at 90%)
        test_idx_set = set(test_idx_fixed)
        remain_positions = [i for i in range(len(train_targets)) if i not in test_idx_set]
        remain_targets = [train_targets[i] for i in remain_positions]
        remain_dataset_indices = [train_dataset_indices[i] for i in remain_positions]
        
        # split directly represents what proportion of training candidate to use (split proportion of 90%)
        train_rel_idx, _ = balanced_stratified_split_indices(
            remain_targets, remain_dataset_indices, split, config.seed
        )
        train_idx = [remain_positions[i] for i in train_rel_idx]
        test_idx = test_idx_fixed
        
        
        tgt_train_subset = Subset(tgt_train_full, train_idx)
        tgt_test_subset = Subset(tgt_test_full, test_idx)
        
        # Count sample distribution of each dataset in training set
        train_dataset_distribution = {}
        for abs_idx in train_idx:
            dataset_idx = train_dataset_indices[abs_idx]
            dataset_name = merged_target_roots[dataset_idx]
            if dataset_name not in train_dataset_distribution:
                train_dataset_distribution[dataset_name] = 0
            train_dataset_distribution[dataset_name] += 1
        logger.info(f"Training set sample distribution: {train_dataset_distribution}")
        
    else:
        # Single dataset case
        tgt_root = datasets_map[target_name]
        tgt_train_ds = load_imagefolder_with_mapping(tgt_root, train_tf, classes, class_to_idx)
        tgt_train_subset = Subset(tgt_train_ds, train_idx)
        tgt_test_subset = Subset(tgt_full_ds, test_idx)

    train_loader = DataLoader(tgt_train_subset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=(device.startswith("cuda")))
    test_loader = DataLoader(tgt_test_subset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=(device.startswith("cuda")))

    # === Build model and do source domain pretraining if needed ===
    model = build_model(num_classes=len(classes), init_mode=init_mode, device=device)

    if init_mode == "random_then_fixed_src":
        if merged_target_roots and fixed_source_name in merged_target_roots:
            logger.warning(f"fixed_source ({fixed_source_name}) is in target datasets, should be 'new dataset' as required. Will continue, but this is more like pretraining and then fine-tuning on same domain.")
        model = pretrain_on_fixed_source(
            model=model,
            source_root=datasets_map[fixed_source_name],
            config=config,
            classes=classes,
            class_to_idx=class_to_idx
        )

    # === Target domain fine-tuning (or train from scratch) ===
    if len(train_idx) == 0:
        logger.info("Fine-tuning ratio is 0, skip training, directly test random/pretrained weights' zero-shot performance on target domain.")
    else:
        target_desc = f"merged datasets {merged_target_roots}" if merged_target_roots and len(merged_target_roots) > 1 else target_name
        logger.info(f"Fine-tuning on target set {target_desc}: ratio {split:.2f} | training samples {len(train_idx)} | test samples {len(test_idx)}")
        logger.info(f"Training epochs: {max_epochs}, test points: {epochs_finetune_list}")
        
        # Prepare base experiment info
        exp_info_base = {
            "target_datasets": merged_target_roots if merged_target_roots else [target_name],
            "split": split,
            "init_mode": init_mode,
            "source_dataset": fixed_source_name if init_mode == "random_then_fixed_src" else "None",
            "epochs_pretrain": config.epochs_pretrain,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "img_size": config.img_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "use_amp": config.use_amp
        }
        
        # Use modified train_one_stage function to test at specified epoch points
        train_results = train_one_stage(
            model, train_loader, 
            epochs=max_epochs, lr=config.lr, device=device, use_amp=config.use_amp,
            test_loader=test_loader, 
            epochs_finetune_list=epochs_finetune_list,
            out_dir=out_dir, 
            exp_info_base=exp_info_base
        )
        
        # logger.info(f"Training completed! Test results saved to {len(train_results.get('test_results', []))} files")

    # === Final testing (if not already tested during training) ===
    # Check if all required test results have been saved during training
    if len(train_idx) > 0 and 'test_results' in train_results:
        saved_epochs = [result['epoch'] for result in train_results['test_results']]
        missing_epochs = [ep for ep in epochs_finetune_list if ep not in saved_epochs]
        
        if missing_epochs:
            logger.info(f"Missing test results for following epochs, supplementing: {missing_epochs}")
            for ep in missing_epochs:
                logger.info(f"Supplementing test for epoch {ep}")
                metrics = evaluate(model, test_loader, device=device, return_report=True)
                
                # Save test results
                exp_info = exp_info_base.copy()
                exp_info.update({
                    "epochs_finetune": ep,
                    **metrics
                })

                # Filter out unnecessary training config parameters
                filtered_info = {
                    "target_datasets": exp_info.get("target_datasets"),
                    "split": exp_info.get("split"),
                    "init_mode": exp_info.get("init_mode"),
                    "source_dataset": exp_info.get("source_dataset"),
                    "epochs_finetune": exp_info.get("epochs_finetune"),
                    "acc": exp_info.get("acc"),
                    "recall": exp_info.get("recall"),
                    "precision": exp_info.get("precision"),
                    "f1_macro": exp_info.get("f1_macro"),
                    "f1_micro": exp_info.get("f1_micro"),
                    "auroc": exp_info.get("auroc"),
                    "cls_report": exp_info.get("cls_report")
                }
                
                # Generate filename
                if merged_target_roots and len(merged_target_roots) > 1:
                    target_name_for_file = "_".join(merged_target_roots)
                else:
                    target_name_for_file = target_name
                out_json = Path(out_dir) / f"{exp_info_base['source_dataset']}_to_{target_name_for_file}_{init_mode}_r{str(split).replace('.', '')}_e{ep}.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(filtered_info, f, ensure_ascii=False, indent=2)
                logger.info(f"Supplementary test results saved: {out_json}")
        else:
            logger.info(f"All epoch test results are complete")
            # Create a base exp_info for return
            exp_info = exp_info_base.copy()
            exp_info.update({
                "epochs_finetune": epochs_finetune_list[0] if epochs_finetune_list else 0,
                "message": "All test results saved during training"
            })
    else:
        # If no training process, directly test
        logger.info("Directly testing...")
        metrics = evaluate(model, test_loader, device=device, return_report=True)
        exp_info = {
            "target_datasets": merged_target_roots if merged_target_roots else [target_name],
            "split": split,
            "init_mode": init_mode,
            "source_dataset": fixed_source_name if init_mode == "random_then_fixed_src" else "None",
            "epochs_finetune": epochs_finetune_list[0] if epochs_finetune_list else 0,
            "epochs_pretrain": config.epochs_pretrain,
            **metrics
        }

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Generate filename
        if merged_target_roots and len(merged_target_roots) > 1:
            target_name_for_file = "_".join(merged_target_roots)
        else:
            target_name_for_file = target_name
        out_json = Path(out_dir) / f"{exp_info_base['source_dataset']}_to_{target_name_for_file}_{init_mode}_r{str(split).replace('.', '')}_e{epochs_finetune_list[0] if epochs_finetune_list else 0}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(exp_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved: {out_json}")
        logger.info(metrics.get("cls_report", ""))

    return exp_info


def generate_summary_from_json_files(out_dir: str, seed: int = 42) -> Optional[Path]:
    """
    Generate summary.csv by reading json files in output directory
    Optimization: use pandas batch operations to improve efficiency
    """
    import pandas as pd
    
    out_path = Path(out_dir)
    if not out_path.exists():
        logger.error(f"Output directory does not exist: {out_dir}")
        return None
    
    # Find all *.json files
    json_files = list(out_path.glob("*.json"))
    if not json_files:
        logger.error(f"No *.json files found in directory {out_dir}")
        return None
    
    logger.info(f"Found {len(json_files)} result files")
    
    # Batch read JSON files
    all_results = []
    failed_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_results.append(data)
            logger.debug(f"Successfully read: {json_file.name}")
        except Exception as e:
            logger.warning(f"Error reading file {json_file}: {e}")
            failed_files.append(json_file.name)
            continue
    
    if not all_results:
        logger.error(f"No result files successfully read")
        return None
    
    if failed_files:
        logger.warning(f"{len(failed_files)} files failed to read: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by specified order: target_datasets -> split -> epochs_finetune
    if all(col in df.columns for col in ['target_datasets', 'init_mode', 'split', 'epochs_finetune']):
        # Process target_datasets column (may be list format)
        def extract_target_name(target_datasets):
            if isinstance(target_datasets, list):
                return '_'.join(target_datasets) if len(target_datasets) > 1 else target_datasets[0]
            else:
                return str(target_datasets)
        
        # Create temporary column for sorting
        df['_target_sort'] = df['target_datasets'].apply(extract_target_name)
        
        # Sort by specified order
        df = df.sort_values(['_target_sort', 'init_mode','split', 'epochs_finetune'], 
                           ascending=[True, True, True, True])
        
        # Delete temporary column
        df = df.drop('_target_sort', axis=1)
        
        logger.info(f"Sorted by target_datasets -> split -> epochs_finetune")
    else:
        logger.warning("Missing columns required for sorting, skipping sort")
    
    # Generate summary.csv
    csv_path = "summary.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Successfully generated summary file: {csv_path}")
    logger.info(f"Total experiments: {len(df)}")
    
    # Show statistics
    logger.info(f"Experiment results statistics:")
    
    # Group by initialization mode
    if 'init_mode' in df.columns:
        logger.info(f"  - By initialization mode:")
        mode_counts = df['init_mode'].value_counts()
        for mode, count in mode_counts.items():
            logger.info(f"    {mode}: {count} experiments")
    
    # Group by fine-tuning epochs
    if 'epochs_finetune' in df.columns:
        logger.info(f"  - By fine-tuning epochs:")
        epoch_counts = df['epochs_finetune'].value_counts()
        for epoch, count in epoch_counts.items():
            logger.info(f"    {epoch} epochs: {count} experiments")
    
    # Group by fine-tuning ratio
    if 'split' in df.columns:
        logger.info(f"  - By fine-tuning ratio:")
        ratio_counts = df['split'].value_counts()
        for ratio, count in ratio_counts.items():
            logger.info(f"    ratio {ratio}: {count} experiments")
    
    # Show CSV file column info
    logger.info(f"CSV file column info:")
    logger.info(f"  Number of columns: {len(df.columns)}")
    logger.info(f"  Column names: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"⚠️  Missing values found:")
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                logger.warning(f"    {col}: {missing_count} missing values")
    else:
        logger.info(f"✓ No missing values")
    
    return csv_path


def run_grid(
        datasets_map: Dict[str, str],
        splits: List[float],
        init_modes: List[str],
        fixed_source_name: str,
        out_dir: str,
        epochs_finetune_list: List[int] = [10, 20, 30],
        epochs_pretrain: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
        seed: int = 42,
        use_amp: bool = True,
        val_split: float = 0.2,
        patience: int = 5,
        target_mode: str = "merged",
) -> List[Dict]:
    """
    Run grid search experiments
    Optimization: use config class, reduce parameter passing
    """
    all_results = []
    
    # Automatically determine target datasets: the other two datasets besides fixed_source
    available_datasets = list(datasets_map.keys())
    if fixed_source_name in available_datasets:
        available_datasets.remove(fixed_source_name)
    
    # Decide whether to merge or transfer separately based on target mode
    if target_mode == "merged":
        target_specs = [{
            "target_name": "_".join(available_datasets),
            "merged_target_roots": available_datasets
        }]
    else:
        target_specs = [{"target_name": name, "merged_target_roots": None} for name in available_datasets]
    
    # Show experiment configuration info
    total_experiments = len(init_modes) * len(splits) * len(target_specs)
    logger.info(f"Experiment configuration:")
    logger.info(f"  - Initialization modes: {init_modes}")
    logger.info(f"  - Fine-tuning ratios: {splits}")
    logger.info(f"  - Fine-tuning epochs: {epochs_finetune_list}")
    logger.info(f"  - Pretraining epochs: {epochs_pretrain}")
    logger.info(f"  - Fixed source dataset: {fixed_source_name}")
    logger.info(f"  - Target mode: {target_mode}")
    logger.info(f"  - Target datasets: {available_datasets if target_mode == 'separate' else available_datasets}")
    logger.info(f"  - Total experiments: {total_experiments} (each experiment will generate {len(epochs_finetune_list)} epoch results)")
    logger.info(f"Starting experiments...")
    
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    for init_mode in init_modes:
        for s in splits:
            for spec in target_specs:
                logger.info(f"Starting experiment: init_mode={init_mode}, split={s}, target={spec['target_name']}")
                logger.info(f"Will test at following epoch points in one training: {epochs_finetune_list}")

                # Check if all target JSONs already exist before running, skip if they do
                expected_files = [
                    Path(out_dir) / build_result_filename(
                        init_mode=init_mode,
                        fixed_source_name=fixed_source_name,
                        merged_target_roots=spec['merged_target_roots'],
                        target_name=spec['target_name'],
                        split=s,
                        epoch=ep
                    )
                    for ep in epochs_finetune_list
                ]
                if all(p.exists() for p in expected_files):
                    logger.info(f"All output files already exist, skipping this sub-experiment: init_mode={init_mode}, split={s}, target={spec['target_name']}")
                    continue
                
                res = run_single_experiment(
                    datasets_map=datasets_map,
                    target_name=spec['target_name'],
                    split=s,
                    init_mode=init_mode,
                    fixed_source_name=fixed_source_name,
                    out_dir=out_dir,
                    epochs_finetune_list=epochs_finetune_list,
                    epochs_pretrain=epochs_pretrain,
                    lr=lr,
                    batch_size=batch_size,
                    img_size=img_size,
                    num_workers=num_workers,
                    seed=seed,
                    use_amp=use_amp,
                    merged_target_roots=spec['merged_target_roots'],
                )
                all_results.append(res)

    # Show experiment results statistics
    logger.info(f"Experiment results statistics:")
    logger.info(f"  - Total experiments: {len(all_results)}")
    logger.info(f"  - By initialization mode:")
    for mode in init_modes:
        mode_count = sum(1 for res in all_results if res.get('init_mode') == mode)
        logger.info(f"    {mode}: {mode_count} experiments")
    
    logger.info(f"  - Each experiment will generate {len(epochs_finetune_list)} epoch result files")
    logger.info(f"All experiment results saved as JSON files to directory: {out_dir}")
    
    return all_results


# ============== CLI ==============
def parse_args():
    """Parse command line arguments (use default values from config file)"""
    p = argparse.ArgumentParser(description="31-class transfer learning batch experiment framework")

    p.add_argument("--fixed_source", type=str, default=CLIDefaults.FIXED_SOURCE,
                   choices=DatasetConfig.AVAILABLE_DATASETS,
                   help="Fixed source dataset for init_mode=random_then_fixed_src")

    p.add_argument("--splits", type=float, nargs="+", default=CLIDefaults.SPLITS,
                   help="List of training candidate usage ratios (training candidate fixed at 90%% of total data / test set fixed at 10%%). For example, 0.9 means use 90%% of training candidate for training")
    p.add_argument("--init_modes", type=str, nargs="+",
                   default=CLIDefaults.INIT_MODES,
                   choices=ModelConfig.INIT_MODES,
                   help="List of initialization/pretraining modes")

    p.add_argument("--epochs_finetune", type=int, nargs="+", default=CLIDefaults.EPOCHS_FINETUNE, 
                   help="List of target domain fine-tuning epochs (default: %(default)s)")
    p.add_argument("--epochs_pretrain", type=int, default=CLIDefaults.EPOCHS_PRETRAIN,
                   help="Pretraining epochs (default: %(default)s)")
    p.add_argument("--val_split", type=float, default=CLIDefaults.VAL_SPLIT, 
                   help="Pretraining validation split ratio (default: %(default)s)")
    p.add_argument("--patience", type=int, default=CLIDefaults.PATIENCE, 
                   help="Early stopping patience value (default: %(default)s)")
    p.add_argument("--batch_size", type=int, default=CLIDefaults.BATCH_SIZE,
                   help="Batch size (default: %(default)s)")
    p.add_argument("--lr", type=float, default=CLIDefaults.LR,
                   help="Learning rate (default: %(default)s)")
    p.add_argument("--img_size", type=int, default=CLIDefaults.IMG_SIZE,
                   help="Image size (default: %(default)s)")
    p.add_argument("--num_workers", type=int, default=CLIDefaults.NUM_WORKERS,
                   help="Number of data loading threads (default: %(default)s)")
    p.add_argument("--seed", type=int, default=CLIDefaults.SEED,
                   help="Random seed (default: %(default)s)")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    p.add_argument("--out_dir", type=str, default=OutputConfig.DEFAULT_OUTPUT_DIR,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--summary", action="store_true", 
                   help="Only generate summary file, don't run experiments (for generating summary from existing json files)")
    p.add_argument("--target_mode", type=str, default=CLIDefaults.TARGET_MODE,
                   choices=["merged", "separate"],
                   help="Target mode: merged (merge two targets) or separate (transfer to two targets separately)")
    return p.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    if args.summary:
        # Only generate summary file
        logger.info("Summary file generation mode only")
        csv_path = generate_summary_from_json_files(args.out_dir, args.seed)
        if csv_path:
            logger.info(f"Summary file generated successfully: {csv_path}")
        else:
            logger.error(f"Summary file generation failed")
        return
    
    # Normal experiment running mode
    seed_all(args.seed)

    # Load dataset mapping from config file
    datasets_map = DatasetConfig.DATASETS_MAP
    logger.info(f"Loaded dataset config: {list(datasets_map.keys())}")
    logger.info(f"Using fine-tuning epochs list: {args.epochs_finetune}")
    
    run_grid(
        datasets_map=datasets_map,
        splits=args.splits,
        init_modes=args.init_modes,
        fixed_source_name=args.fixed_source,
        out_dir=args.out_dir,
        epochs_finetune_list=args.epochs_finetune,
        epochs_pretrain=args.epochs_pretrain,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
        val_split=args.val_split,
        patience=args.patience,
        target_mode=args.target_mode,
    )


if __name__ == "__main__":
    main()
