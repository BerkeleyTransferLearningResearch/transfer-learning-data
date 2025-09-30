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

# 导入配置文件
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

# 线程锁用于文件写入
file_lock = threading.Lock()

# 配置日志
logging.basicConfig(level=logging.INFO, format=OutputConfig.LOG_FORMAT)
logger = logging.getLogger(__name__)

# 为了向后兼容，保留这些常量
IMAGENET_MEAN = ImageConfig.IMAGENET_MEAN
IMAGENET_STD = ImageConfig.IMAGENET_STD

# ============== 基础配置 ==============
def seed_all(seed: int = 42):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 CuDNN 更可复现（会略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """构建训练、增强训练和测试数据变换"""
    
    # 增强训练变换 - 更激进的数据增强
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
    返回一个 ImageFolder，并确保 class_to_idx 的顺序与 classes（若给定）一致。
    不给 classes 时，按当前 root 的类名（排序）作为参考。
    """
    # 先用一次 ImageFolder 看看类名
    tmp = datasets.ImageFolder(root)
    cur_classes = sorted(tmp.class_to_idx.keys())
    if classes is None:
        classes = cur_classes
    else:
        # 校验类集合一致
        if set(cur_classes) != set(classes):
            raise ValueError(f"数据集 {root} 的类与参考不一致：\n"
                             f"参考: {sorted(classes)}\n当前: {sorted(cur_classes)}")



    # 强制使用参考 classes 的顺序构造映射
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def load_imagefolder_with_mapping(root: str, transform, classes: List[str], class_to_idx: Dict[str, int]) -> datasets.ImageFolder:
    """加载ImageFolder并重新映射类别"""
    ds = datasets.ImageFolder(root=root, transform=transform)
    # 重新映射 targets 到统一的 class_to_idx
    # 注意：ImageFolder 内部使用 samples(list of (path, target))，这里重写它们的 target
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
    对 ImageFolder 的 targets 做分层抽样，返回 (train_idx, test_idx)
    ratio 表示训练(微调)比例。若 ratio==0 则训练集为空。
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    train_idx, test_idx = [], []
    classes = np.unique(targets)
    for c in classes:
        idxs = np.where(targets == c)[0]
        rng.shuffle(idxs)
        k = int(round(len(idxs) * ratio))
        # 允许 k == 0（即该类不参与微调），保持严格的"剩余为测试"语义
        train_idx.extend(idxs[:k].tolist())
        test_idx.extend(idxs[k:].tolist())
    # 只打乱训练集，保持测试集顺序以确保结果一致性
    rng.shuffle(train_idx)
    return train_idx, test_idx


def balanced_stratified_split_indices(targets: List[int], dataset_indices: List[int], ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    平衡分层抽样：确保每个类别中，来自不同数据集的样本数量相等
    targets: 样本的类别标签
    dataset_indices: 样本对应的数据集索引（0表示第一个数据集，1表示第二个数据集）
    ratio: 训练比例
    返回: (train_idx, test_idx)
    """
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    dataset_indices = np.array(dataset_indices)
    
    train_idx, test_idx = [], []
    classes = np.unique(targets)
    
    for c in classes:
        # 找到该类别的所有样本
        class_mask = targets == c
        class_indices = np.where(class_mask)[0]
        
        # 按数据集分组
        dataset1_indices = class_indices[dataset_indices[class_indices] == 0]  # 第一个数据集
        dataset2_indices = class_indices[dataset_indices[class_indices] == 1]  # 第二个数据集
        
        # 计算每个数据集应该选择的训练样本数
        n_train_d1 = int(round(len(dataset1_indices) * ratio))
        n_train_d2 = int(round(len(dataset2_indices) * ratio))
        
        # 随机选择
        rng.shuffle(dataset1_indices)
        rng.shuffle(dataset2_indices)
        
        train_idx.extend(dataset1_indices[:n_train_d1].tolist())
        train_idx.extend(dataset2_indices[:n_train_d2].tolist())
        test_idx.extend(dataset1_indices[n_train_d1:].tolist())
        test_idx.extend(dataset2_indices[n_train_d2:].tolist())
    
    # 只打乱训练集，保持测试集顺序以确保结果一致性
    rng.shuffle(train_idx)
    
    return train_idx, test_idx



# ============== 模型相关 ==============
def build_model(num_classes: int,
                init_mode: str = "resnet50_imagenet",
                device: str = "cuda") -> nn.Module:
    """
    构建模型
    init_mode:
      - "resnet50_imagenet"      : 完全 resnet50（ImageNet 预训练）
      - "random"                 : 完全随机权重
      - "random_then_fixed_src"  : 完全随机 -> 在固定源数据上监督预训练
    """
    if init_mode == "resnet50_imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif init_mode in ["random", "random_then_fixed_src"]:
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"未知 init_mode: {init_mode}")

    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model.to(device)


class MetricsCalculator:
    """指标计算器"""
    def __init__(self):
        self.cache = {}
    
    def calculate_metrics(self, all_true: List[int], all_pred: List[int], cache_key: str = None) -> Dict:
        """计算所有指标"""
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
    y_score=None,                 # 新增：用于计算每类 ovr-AUC
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=False,
    zero_division="warn",
    per_class_accuracy="balanced",  # "balanced" | "recall" | "specificity"
):
    # 解析标签与名称
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    if target_names is None:
        target_names = [str(l) for l in labels]

    # 原有三项指标
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    # 每类混淆矩阵 (一对多)
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

    # 每类 ovr-AUC
    aucs = np.full(len(labels), np.nan, dtype=float)
    if y_score is not None:
        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        if y_score_arr.ndim == 1:  # 二分类 y_score 为正类分数
            # 找到“正类”标签（默认 greater label）
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
            # 多分类：逐类一对多
            for i, lab in enumerate(labels):
                bin_true = (y_true_arr == lab).astype(int)
                try:
                    aucs[i] = roc_auc_score(bin_true, y_score_arr[:, i])
                except Exception:
                    aucs[i] = np.nan
    # 构建输出
    headers = ["precision", "recall", "f1-score", "support", "specificity", "balanced-acc", "ovr-auc"]
    rows = zip(target_names, p, r, f1, s, specificity, per_acc, aucs)

    if output_dict:
        report = {}
        for row in rows:
            name, *vals = row
            report[name] = dict(zip(headers, [float(v) if v is not None and not np.isnan(v) else np.nan for v in vals]))
        # 追加宏/加权/微平均（保持与原报告一致；新列按 NaN 填充）
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
        # 文本格式
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

        # 追加 accuracy / macro / weighted 等汇总（与原输出一致）
        base_txt = sk_classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=False,
            zero_division=zero_division,
        )
        out += base_txt.split("\n", 1)[1]  # 复用原表的汇总部分
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
    训练一个阶段，支持验证集和早停
    优化：减少重复计算，使用更高效的数据处理
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 早停相关变量
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # 测试相关变量
    test_results = []
    if test_loader is not None and epochs_finetune_list is not None:
        logger.info(f"将在以下epoch进行测试: {epochs_finetune_list}")

    # 预分配张量以减少内存分配
    device_obj = torch.device(device)
    
    for ep in range(1, epochs + 1):
        # 训练阶段
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

            # 使用更高效的计算方式
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == y).sum().item()
            total_cnt += batch_size

        train_loss = total_loss / max(1, total_cnt)
        train_acc = total_correct / max(1, total_cnt)
        
        # 验证阶段
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
            
            # 早停检查
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Epoch {ep}/{epochs} | 早停触发！验证集准确率连续{patience}个epoch未提升")
                # 恢复最佳模型状态
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        # 在指定epoch进行测试
        if test_loader is not None and epochs_finetune_list is not None and ep in epochs_finetune_list:
            logger.info(f"Epoch {ep}/{epochs} | 开始测试...")
            test_metrics = evaluate(model, test_loader, device=device, return_report=True)
            
            # 异步保存测试结果
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
    """异步保存测试结果"""
    def save_worker():
        try:
            test_exp_info = exp_info_base.copy()
            test_exp_info.update({
                "epochs_finetune": ep,
                **test_metrics
            })
            
            # 过滤掉不需要的训练配置参数
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
            
            # 生成文件名
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
            
            logger.info(f"Epoch {ep} 测试结果已保存：{out_json}")
            
            # 记录测试结果
            test_results.append({
                "epoch": ep,
                "metrics": test_metrics,
                "file_path": str(out_json)
            })
        except Exception as e:
            logger.error(f"保存测试结果时出错: {e}")
    
    # 使用线程池异步执行
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(save_worker)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cuda",
             return_report: bool = True) -> Dict:
    """评估模型性能，优化内存使用"""
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

    # 使用指标计算器
    metrics_calc = MetricsCalculator()
    out = metrics_calc.calculate_metrics(all_true, all_pred)
    
    # 计算多分类 AUROC（macro，OVR），在类别不全或分数异常时回退为 None
    try:
        auroc = roc_auc_score(np.array(all_true), np.array(all_scores), multi_class="ovr", average="macro")
        out["auroc"] = float(auroc)
    except Exception:
        out["auroc"] = None
    
    if return_report:
        out["cls_report"] = classification_report_plus(all_true, all_pred, y_score=np.array(all_scores), digits=4, zero_division=0)
    
    return out


# ============== 实验管线 ==============
def pretrain_on_fixed_source(model: nn.Module,
                             source_root: str,
                             config: ExperimentConfig,
                             classes: List[str],
                             class_to_idx: Dict[str, int]) -> nn.Module:
    """
    用于 init_mode == random_then_fixed_src 的监督预训练阶段。
    支持验证集分割和早停机制。
    """
    train_tf, test_tf = build_transforms(config.img_size)
    
    # 加载源数据集
    src_ds = load_imagefolder_with_mapping(source_root, train_tf, classes, class_to_idx)
    
    # 分割训练集和验证集
    total_samples = len(src_ds)
    val_size = int(total_samples * config.val_split)
    train_size = total_samples - val_size
    
    # 使用分层抽样确保类别平衡
    targets = src_ds.targets
    train_idx, val_idx = stratified_split_indices(targets, 1 - config.val_split, seed=config.seed)
    
    # 创建训练和验证子集
    train_subset = Subset(src_ds, train_idx)
    val_subset = Subset(src_ds, val_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=(config.use_amp))
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=(config.use_amp))
    
    logger.info(f"在固定源数据集 ({source_root}) 上进行监督预训练")
    logger.info(f"训练集: {len(train_subset)} 样本, 验证集: {len(val_subset)} 样本")
    logger.info(f"总样本数: {total_samples}, 验证集比例: {config.val_split:.1%}")
    
    # 使用带验证集的训练
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
    
    logger.info(f"预训练完成！最佳验证准确率: {train_results['best_val_acc']:.4f}")
    logger.info(f"实际训练轮数: {train_results['epochs_trained']}/{config.epochs_pretrain}")
    
    return model


# 继续优化其他函数...

def build_result_filename(init_mode: str,
                          fixed_source_name: str,
                          merged_target_roots: Optional[List[str]],
                          target_name: str,
                          split: float,
                          epoch: int) -> str:
    """构造结果文件名（与保存逻辑一致）"""
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
    针对一个 target + split + init_mode 的完整实验：
    优化：使用配置类，减少重复代码，提高可读性
    """
    logger.debug(f"开始实验: target={target_name}, split={split}, init_mode={init_mode}")
    logger.info(f"数据集映射: {list(datasets_map.keys())}")
    if merged_target_roots:
        logger.info(f"合并目标数据集: {merged_target_roots}")
    
    # 创建配置对象
    config = ExperimentConfig(
        batch_size=batch_size,
        lr=lr,
        img_size=img_size,
        num_workers=num_workers,
        seed=seed,
        use_amp=use_amp,
        epochs_pretrain=epochs_pretrain
    )
    
    # 处理epochs_finetune参数
    if epochs_finetune_list is not None:
        max_epochs = max(epochs_finetune_list)
        logger.info(f"使用多个微调轮数: {epochs_finetune_list}，最大训练轮数: {max_epochs}")
    else:
        max_epochs = epochs_finetune
        epochs_finetune_list = [epochs_finetune]
        logger.info(f"使用单个微调轮数: {epochs_finetune}")
    
    seed_all(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 统一类映射 ===
    names = list(datasets_map.keys())
    first_root = datasets_map[names[0]]
    classes, class_to_idx = enforce_class_mapping(first_root, classes=None)
    # 校验另外两个
    for k, root in datasets_map.items():
        enforce_class_mapping(root, classes=classes)

    # === 数据集准备 ===
    train_tf, test_tf = build_transforms(config.img_size)

    # 目标数据集：如果是合并数据集，则合并多个数据集
    if merged_target_roots and len(merged_target_roots) > 1:
        # 合并多个数据集 - 使用 train_tf 确保一致性
        tgt_datasets = []
        for dataset_name in merged_target_roots:
            root = datasets_map[dataset_name]
            ds = load_imagefolder_with_mapping(root, train_tf, classes, class_to_idx)
            tgt_datasets.append(ds)
        
        # 合并数据集
        tgt_full_ds = ConcatDataset(tgt_datasets)
        
        # 重新计算 targets 列表
        all_targets = []
        for ds in tgt_datasets:
            if hasattr(ds, 'targets'):
                all_targets.extend(ds.targets)
            else:
                all_targets.extend([y for _, y in ds.samples])
        
        logger.info(f"合并目标数据集：{merged_target_roots}，总样本数：{len(tgt_full_ds)}")
    else:
        # 单个数据集的情况
        tgt_root = datasets_map[target_name]
        tgt_full_ds = load_imagefolder_with_mapping(tgt_root, test_tf, classes, class_to_idx)
        all_targets = tgt_full_ds.targets

        # 固定测试集为10%，训练集候选为90%，split决定使用多少比例的训练集候选
    # 先确定固定测试集索引（10%）
    _, test_idx = stratified_split_indices(all_targets, 1 - 0.1, config.seed)
    # 训练候选集：除去测试集后的样本（固定为90%）
    all_indices = list(range(len(all_targets)))
    test_idx_set = set(test_idx)
    remain_indices = [i for i in all_indices if i not in test_idx_set]
    # split 直接表示使用多少比例的训练集候选（90%中的split比例）
    remain_targets = [all_targets[i] for i in remain_indices]
    train_rel_idx, _ = stratified_split_indices(remain_targets, split, config.seed)
    train_idx = [remain_indices[i] for i in train_rel_idx]
    
    train_ratio_of_total = len(train_idx) / len(all_targets) if len(all_targets) > 0 else 0
    logger.info(f"固定测试比例10% | 训练集候选90% | 使用{split*100:.0f}%训练集候选 | 训练样本 {len(train_idx)} ({train_ratio_of_total*100:.1f}%总数据) | 测试样本 {len(test_idx)}")
              
    # 创建训练和测试子集
    if merged_target_roots and len(merged_target_roots) > 1:
        # 对于合并数据集，需要重新构建训练和测试数据
        train_datasets = []
        test_datasets = []
        
        # 重新加载每个数据集并应用相应的 transform
        for dataset_name in merged_target_roots:
            root = datasets_map[dataset_name]
            train_ds = load_imagefolder_with_mapping(root, train_tf, classes, class_to_idx)
            test_ds = load_imagefolder_with_mapping(root, test_tf, classes, class_to_idx)
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        
        # 合并训练和测试数据集
        tgt_train_full = ConcatDataset(train_datasets)
        tgt_test_full = ConcatDataset(test_datasets)
        
        # 重新计算 targets 和 dataset_indices（两组的顺序应一致）
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
        
        # 先固定测试集为10%（对每个数据集、每个类别分别取10%），保持跨split不变
        fixed_test_ratio = 0.1
        _, test_idx_fixed = balanced_stratified_split_indices(
            test_targets, test_dataset_indices, 1 - fixed_test_ratio, config.seed
        )
        
        # 训练候选集：训练合并集里去除测试索引对应位置（固定为90%）
        test_idx_set = set(test_idx_fixed)
        remain_positions = [i for i in range(len(train_targets)) if i not in test_idx_set]
        remain_targets = [train_targets[i] for i in remain_positions]
        remain_dataset_indices = [train_dataset_indices[i] for i in remain_positions]
        
        # split 直接表示使用多少比例的训练集候选（90%中的split比例）
        train_rel_idx, _ = balanced_stratified_split_indices(
            remain_targets, remain_dataset_indices, split, config.seed
        )
        train_idx = [remain_positions[i] for i in train_rel_idx]
        test_idx = test_idx_fixed
        
        
        tgt_train_subset = Subset(tgt_train_full, train_idx)
        tgt_test_subset = Subset(tgt_test_full, test_idx)
        
        # 统计训练集中每个数据集的样本分布
        train_dataset_distribution = {}
        for abs_idx in train_idx:
            dataset_idx = train_dataset_indices[abs_idx]
            dataset_name = merged_target_roots[dataset_idx]
            if dataset_name not in train_dataset_distribution:
                train_dataset_distribution[dataset_name] = 0
            train_dataset_distribution[dataset_name] += 1
        logger.info(f"训练集样本分布: {train_dataset_distribution}")
        
    else:
        # 单个数据集的情况
        tgt_root = datasets_map[target_name]
        tgt_train_ds = load_imagefolder_with_mapping(tgt_root, train_tf, classes, class_to_idx)
        tgt_train_subset = Subset(tgt_train_ds, train_idx)
        tgt_test_subset = Subset(tgt_full_ds, test_idx)

    train_loader = DataLoader(tgt_train_subset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=(device.startswith("cuda")))
    test_loader = DataLoader(tgt_test_subset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=(device.startswith("cuda")))

    # === 构建模型并视情况做源域预训练 ===
    model = build_model(num_classes=len(classes), init_mode=init_mode, device=device)

    if init_mode == "random_then_fixed_src":
        if merged_target_roots and fixed_source_name in merged_target_roots:
            logger.warning(f"fixed_source ({fixed_source_name}) 在目标数据集中，按要求本应为'新数据集'。将继续执行，但这更像是在同一域预训练后再微调。")
        model = pretrain_on_fixed_source(
            model=model,
            source_root=datasets_map[fixed_source_name],
            config=config,
            classes=classes,
            class_to_idx=class_to_idx
        )

    # === 目标域微调（或从零训练） ===
    if len(train_idx) == 0:
        logger.info("微调比例为 0，跳过训练，直接测试随机/预训练权重在目标域的零样本表现。")
    else:
        target_desc = f"合并数据集 {merged_target_roots}" if merged_target_roots and len(merged_target_roots) > 1 else target_name
        logger.info(f"在目标集 {target_desc} 上微调：比例 {split:.2f} | 训练样本 {len(train_idx)} | 测试样本 {len(test_idx)}")
        logger.info(f"训练轮数: {max_epochs}, 测试点: {epochs_finetune_list}")
        
        # 准备基础实验信息
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
        
        # 使用修改后的train_one_stage函数，在指定epoch点进行测试
        train_results = train_one_stage(
            model, train_loader, 
            epochs=max_epochs, lr=config.lr, device=device, use_amp=config.use_amp,
            test_loader=test_loader, 
            epochs_finetune_list=epochs_finetune_list,
            out_dir=out_dir, 
            exp_info_base=exp_info_base
        )
        
        # logger.info(f"训练完成！测试结果已保存到 {len(train_results.get('test_results', []))} 个文件")

    # === 最终测试（如果还没有在训练过程中测试过） ===
    # 检查是否已经在训练过程中保存了所有需要的测试结果
    if len(train_idx) > 0 and 'test_results' in train_results:
        saved_epochs = [result['epoch'] for result in train_results['test_results']]
        missing_epochs = [ep for ep in epochs_finetune_list if ep not in saved_epochs]
        
        if missing_epochs:
            logger.info(f"以下epoch的测试结果缺失，进行补充测试: {missing_epochs}")
            for ep in missing_epochs:
                logger.info(f"补充测试 epoch {ep}")
                metrics = evaluate(model, test_loader, device=device, return_report=True)
                
                # 保存测试结果
                exp_info = exp_info_base.copy()
                exp_info.update({
                    "epochs_finetune": ep,
                    **metrics
                })

                # 过滤掉不需要的训练配置参数
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
                
                # 生成文件名
                if merged_target_roots and len(merged_target_roots) > 1:
                    target_name_for_file = "_".join(merged_target_roots)
                else:
                    target_name_for_file = target_name
                out_json = Path(out_dir) / f"{exp_info_base['source_dataset']}_to_{target_name_for_file}_{init_mode}_r{str(split).replace('.', '')}_e{ep}.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(filtered_info, f, ensure_ascii=False, indent=2)
                logger.info(f"补充测试结果已保存：{out_json}")
        else:
            logger.info(f"所有epoch的测试结果已完整保存")
            # 创建一个基础的exp_info用于返回
            exp_info = exp_info_base.copy()
            exp_info.update({
                "epochs_finetune": epochs_finetune_list[0] if epochs_finetune_list else 0,
                "message": "所有测试结果已在训练过程中保存"
            })
    else:
        # 如果没有训练过程，直接进行测试
        logger.info("直接进行测试...")
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
        # 生成文件名
        if merged_target_roots and len(merged_target_roots) > 1:
            target_name_for_file = "_".join(merged_target_roots)
        else:
            target_name_for_file = target_name
        out_json = Path(out_dir) / f"{exp_info_base['source_dataset']}_to_{target_name_for_file}_{init_mode}_r{str(split).replace('.', '')}_e{epochs_finetune_list[0] if epochs_finetune_list else 0}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(exp_info, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存：{out_json}")
        logger.info(metrics.get("cls_report", ""))

    return exp_info


def generate_summary_from_json_files(out_dir: str, seed: int = 42) -> Optional[Path]:
    """
    通过读取输出目录里的json文件生成summary.csv
    优化：使用pandas的批量操作，提高效率
    """
    import pandas as pd
    
    out_path = Path(out_dir)
    if not out_path.exists():
        logger.error(f"输出目录不存在: {out_dir}")
        return None
    
    # 查找所有*.json文件
    json_files = list(out_path.glob("*.json"))
    if not json_files:
        logger.error(f"在目录 {out_dir} 中未找到任何*.json文件")
        return None
    
    logger.info(f"找到 {len(json_files)} 个结果文件")
    
    # 批量读取JSON文件
    all_results = []
    failed_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_results.append(data)
            logger.debug(f"成功读取: {json_file.name}")
        except Exception as e:
            logger.warning(f"读取文件 {json_file} 时出错: {e}")
            failed_files.append(json_file.name)
            continue
    
    if not all_results:
        logger.error(f"没有成功读取任何结果文件")
        return None
    
    if failed_files:
        logger.warning(f"{len(failed_files)} 个文件读取失败: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 按指定顺序排序：target_datasets -> split -> epochs_finetune
    if all(col in df.columns for col in ['target_datasets', 'init_mode', 'split', 'epochs_finetune']):
        # 处理target_datasets列（可能是列表格式）
        def extract_target_name(target_datasets):
            if isinstance(target_datasets, list):
                return '_'.join(target_datasets) if len(target_datasets) > 1 else target_datasets[0]
            else:
                return str(target_datasets)
        
        # 创建临时列用于排序
        df['_target_sort'] = df['target_datasets'].apply(extract_target_name)
        
        # 按指定顺序排序
        df = df.sort_values(['_target_sort', 'init_mode','split', 'epochs_finetune'], 
                           ascending=[True, True, True, True])
        
        # 删除临时列
        df = df.drop('_target_sort', axis=1)
        
        logger.info(f"已按 target_datasets -> split -> epochs_finetune 排序")
    else:
        logger.warning("缺少排序所需的列，跳过排序")
    
    # 生成summary.csv
    csv_path = "summary.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"成功生成summary文件: {csv_path}")
    logger.info(f"总实验数: {len(df)}")
    
    # 显示统计信息
    logger.info(f"实验结果统计:")
    
    # 按初始化模式分组
    if 'init_mode' in df.columns:
        logger.info(f"  - 按初始化模式分组:")
        mode_counts = df['init_mode'].value_counts()
        for mode, count in mode_counts.items():
            logger.info(f"    {mode}: {count} 个实验")
    
    # 按微调轮数分组
    if 'epochs_finetune' in df.columns:
        logger.info(f"  - 按微调轮数分组:")
        epoch_counts = df['epochs_finetune'].value_counts()
        for epoch, count in epoch_counts.items():
            logger.info(f"    {epoch} epochs: {count} 个实验")
    
    # 按微调比例分组
    if 'split' in df.columns:
        logger.info(f"  - 按微调比例分组:")
        ratio_counts = df['split'].value_counts()
        for ratio, count in ratio_counts.items():
            logger.info(f"    ratio {ratio}: {count} 个实验")
    
    # 显示CSV文件的列信息
    logger.info(f"CSV文件列信息:")
    logger.info(f"  列数: {len(df.columns)}")
    logger.info(f"  列名: {list(df.columns)}")
    
    # 检查是否有缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"⚠️  发现缺失值:")
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                logger.warning(f"    {col}: {missing_count} 个缺失值")
    else:
        logger.info(f"✓ 无缺失值")
    
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
    运行网格搜索实验
    优化：使用配置类，减少参数传递
    """
    all_results = []
    
    # 自动确定目标数据集：除了 fixed_source 之外的另外两个数据集
    available_datasets = list(datasets_map.keys())
    if fixed_source_name in available_datasets:
        available_datasets.remove(fixed_source_name)
    
    # 根据目标模式决定是合并还是分别迁移
    if target_mode == "merged":
        target_specs = [{
            "target_name": "_".join(available_datasets),
            "merged_target_roots": available_datasets
        }]
    else:
        target_specs = [{"target_name": name, "merged_target_roots": None} for name in available_datasets]
    
    # 显示实验配置信息
    total_experiments = len(init_modes) * len(splits) * len(target_specs)
    logger.info(f"实验配置:")
    logger.info(f"  - 初始化模式: {init_modes}")
    logger.info(f"  - 微调比例: {splits}")
    logger.info(f"  - 微调轮数: {epochs_finetune_list}")
    logger.info(f"  - 预训练轮数: {epochs_pretrain}")
    logger.info(f"  - 固定源数据集: {fixed_source_name}")
    logger.info(f"  - 目标模式: {target_mode}")
    logger.info(f"  - 目标数据集: {available_datasets if target_mode == 'separate' else available_datasets}")
    logger.info(f"  - 总实验数: {total_experiments} (每个实验将生成 {len(epochs_finetune_list)} 个epoch的结果)")
    logger.info(f"开始执行实验...")
    
    # 创建输出目录
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {out_dir}")

    for init_mode in init_modes:
        for s in splits:
            for spec in target_specs:
                logger.info(f"开始实验: init_mode={init_mode}, split={s}, target={spec['target_name']}")
                logger.info(f"将在一次训练中测试以下epoch点: {epochs_finetune_list}")

                # 在运行前检查所有目标JSON是否已存在，若已存在则跳过
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
                    logger.info(f"已存在所有输出文件，跳过该子实验: init_mode={init_mode}, split={s}, target={spec['target_name']}")
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

    # 显示实验结果统计
    logger.info(f"实验结果统计:")
    logger.info(f"  - 总实验数: {len(all_results)}")
    logger.info(f"  - 按初始化模式分组:")
    for mode in init_modes:
        mode_count = sum(1 for res in all_results if res.get('init_mode') == mode)
        logger.info(f"    {mode}: {mode_count} 个实验")
    
    logger.info(f"  - 每个实验将生成 {len(epochs_finetune_list)} 个epoch的结果文件")
    logger.info(f"所有实验结果已保存为JSON文件到目录: {out_dir}")
    
    return all_results


# ============== CLI ==============
def parse_args():
    """解析命令行参数（使用配置文件中的默认值）"""
    p = argparse.ArgumentParser(description="31分类迁移学习批量实验框架")

    p.add_argument("--fixed_source", type=str, default=CLIDefaults.FIXED_SOURCE,
                   choices=DatasetConfig.AVAILABLE_DATASETS,
                   help="用于 init_mode=random_then_fixed_src 的固定源数据集")

    p.add_argument("--splits", type=float, nargs="+", default=CLIDefaults.SPLITS,
                   help="使用训练集候选的比例列表(训练集候选固定为总数据的90%%/测试集固定为10%%). 例如0.9表示使用90%%的训练集候选进行训练")
    p.add_argument("--init_modes", type=str, nargs="+",
                   default=CLIDefaults.INIT_MODES,
                   choices=ModelConfig.INIT_MODES,
                   help="初始化/预训练模式列表")

    p.add_argument("--epochs_finetune", type=int, nargs="+", default=CLIDefaults.EPOCHS_FINETUNE, 
                   help="目标域微调轮数列表 (default: %(default)s)")
    p.add_argument("--epochs_pretrain", type=int, default=CLIDefaults.EPOCHS_PRETRAIN,
                   help="预训练轮数 (default: %(default)s)")
    p.add_argument("--val_split", type=float, default=CLIDefaults.VAL_SPLIT, 
                   help="预训练验证集比例 (default: %(default)s)")
    p.add_argument("--patience", type=int, default=CLIDefaults.PATIENCE, 
                   help="早停耐心值 (default: %(default)s)")
    p.add_argument("--batch_size", type=int, default=CLIDefaults.BATCH_SIZE,
                   help="批次大小 (default: %(default)s)")
    p.add_argument("--lr", type=float, default=CLIDefaults.LR,
                   help="学习率 (default: %(default)s)")
    p.add_argument("--img_size", type=int, default=CLIDefaults.IMG_SIZE,
                   help="图像大小 (default: %(default)s)")
    p.add_argument("--num_workers", type=int, default=CLIDefaults.NUM_WORKERS,
                   help="数据加载线程数 (default: %(default)s)")
    p.add_argument("--seed", type=int, default=CLIDefaults.SEED,
                   help="随机种子 (default: %(default)s)")
    p.add_argument("--no_amp", action="store_true", help="关闭混合精度训练")
    p.add_argument("--out_dir", type=str, default=OutputConfig.DEFAULT_OUTPUT_DIR,
                   help="输出目录 (default: %(default)s)")
    p.add_argument("--summary", action="store_true", 
                   help="仅生成summary文件，不运行实验（用于从现有json文件生成summary）")
    p.add_argument("--target_mode", type=str, default=CLIDefaults.TARGET_MODE,
                   choices=["merged", "separate"],
                   help="目标模式: merged(合并两个目标) 或 separate(分别迁移到两个目标)")
    return p.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.summary:
        # 仅生成summary文件
        logger.info("仅生成summary文件模式")
        csv_path = generate_summary_from_json_files(args.out_dir, args.seed)
        if csv_path:
            logger.info(f"Summary文件生成成功: {csv_path}")
        else:
            logger.error(f"Summary文件生成失败")
        return
    
    # 正常运行实验模式
    seed_all(args.seed)

    # 从配置文件加载数据集映射
    datasets_map = DatasetConfig.DATASETS_MAP
    logger.info(f"加载数据集配置: {list(datasets_map.keys())}")
    logger.info(f"使用微调轮数列表: {args.epochs_finetune}")
    
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
