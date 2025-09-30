# config.py
# -*- coding: utf-8 -*-
"""
实验配置文件
包含所有实验相关的配置参数、数据集路径和常量定义
"""
from dataclasses import dataclass
from typing import Dict


# ============== 实验配置类 ==============
@dataclass
class ExperimentConfig:
    """实验配置类"""
    # 训练超参数
    batch_size: int = 32
    lr: float = 1e-3
    img_size: int = 224
    num_workers: int = 4
    seed: int = 42
    
    # 训练策略
    use_amp: bool = True                # 是否使用混合精度训练
    val_split: float = 0.2              # 预训练验证集比例
    patience: int = 5                   # 早停耐心值
    
    # 训练轮数
    epochs_pretrain: int = 50           # 预训练轮数
    epochs_finetune_list: list = None   # 微调轮数列表，例如 [10, 20, 30]
    
    # 数据分割
    fixed_test_ratio: float = 0.1       # 固定测试集比例（10%）
    
    def __post_init__(self):
        """初始化后处理"""
        if self.epochs_finetune_list is None:
            self.epochs_finetune_list = [10, 20, 30]


# ============== 数据集配置 ==============
class DatasetConfig:
    """数据集路径配置"""
    
    # Office-31 数据集路径
    BASE_PATH = '/home/exp/exp/ruih/newdr/Office-31-20250831T042022Z-1-001/Office-31/Data'
    
    DATASETS_MAP = {
        "amazon": f'{BASE_PATH}/amazon',
        "dslr": f'{BASE_PATH}/dslr',
        "webcam": f'{BASE_PATH}/webcam',
    }
    
    # 可用的数据集名称
    AVAILABLE_DATASETS = list(DATASETS_MAP.keys())
    
    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """获取指定数据集的路径"""
        if dataset_name not in cls.DATASETS_MAP:
            raise ValueError(f"未知数据集: {dataset_name}. 可用数据集: {cls.AVAILABLE_DATASETS}")
        return cls.DATASETS_MAP[dataset_name]
    
    @classmethod
    def validate_datasets(cls, dataset_names: list) -> bool:
        """验证数据集名称是否都有效"""
        for name in dataset_names:
            if name not in cls.DATASETS_MAP:
                raise ValueError(f"未知数据集: {name}. 可用数据集: {cls.AVAILABLE_DATASETS}")
        return True


# ============== 模型配置 ==============
class ModelConfig:
    """模型相关配置"""
    
    # 支持的初始化模式
    INIT_MODES = [
        "resnet50_imagenet",      # ResNet50 + ImageNet预训练权重
        "random",                  # 完全随机初始化
        "random_then_fixed_src",   # 随机初始化 + 固定源数据集预训练
    ]
    
    # 模型架构
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 31  # Office-31 数据集有31个类别
    
    @classmethod
    def validate_init_mode(cls, init_mode: str) -> bool:
        """验证初始化模式是否有效"""
        if init_mode not in cls.INIT_MODES:
            raise ValueError(f"未知初始化模式: {init_mode}. 可用模式: {cls.INIT_MODES}")
        return True


# ============== 图像处理常量 ==============
class ImageConfig:
    """图像预处理相关配置"""
    
    # ImageNet 标准化参数
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # 默认图像大小
    DEFAULT_IMG_SIZE = 224
    


# ============== 实验输出配置 ==============
class OutputConfig:
    """实验输出相关配置"""
    
    # 默认输出目录
    DEFAULT_OUTPUT_DIR = "./exp_out"
    
    # 结果文件命名格式
    RESULT_FILENAME_FORMAT = "{source}_to_{target}_{init_mode}_r{split}_e{epoch}.json"
    
    # 汇总文件名
    SUMMARY_FILENAME = "summary.csv"
    
    # 日志格式
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'


# ============== 命令行参数默认值 ==============
class CLIDefaults:
    """命令行参数的默认值"""
    
    # 固定源数据集
    FIXED_SOURCE = "amazon"
    
    # 训练集使用比例（相对于训练集候选）
    SPLITS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    # 初始化模式
    INIT_MODES = ["resnet50_imagenet", "random", "random_then_fixed_src"]
    
    # 微调轮数
    EPOCHS_FINETUNE = [100]
    
    # 预训练轮数
    EPOCHS_PRETRAIN = 50
    
    # 验证集比例
    VAL_SPLIT = 0.2
    
    # 早停耐心值
    PATIENCE = 5
    
    # 训练超参数
    BATCH_SIZE = 32
    LR = 1e-3
    IMG_SIZE = 224
    NUM_WORKERS = 4
    SEED = 42
    
    # 目标模式
    TARGET_MODE = "merged"  # "merged" 或 "separate"


# ============== 全局配置实例 ==============
# 创建默认配置实例
def get_default_config() -> ExperimentConfig:
    """获取默认实验配置"""
    return ExperimentConfig(
        batch_size=CLIDefaults.BATCH_SIZE,
        lr=CLIDefaults.LR,
        img_size=CLIDefaults.IMG_SIZE,
        num_workers=CLIDefaults.NUM_WORKERS,
        seed=CLIDefaults.SEED,
        use_amp=True,
        val_split=CLIDefaults.VAL_SPLIT,
        patience=CLIDefaults.PATIENCE,
        epochs_pretrain=CLIDefaults.EPOCHS_PRETRAIN,
        epochs_finetune_list=CLIDefaults.EPOCHS_FINETUNE,
    )


# ============== 配置验证 ==============
def validate_config(config: ExperimentConfig) -> bool:
    """验证配置是否合理"""
    assert 0 < config.batch_size <= 256, "batch_size 应在 1-256 之间"
    assert 0 < config.lr <= 1.0, "学习率应在 0-1 之间"
    assert config.img_size > 0, "图像大小应为正数"
    assert 0 <= config.val_split < 1.0, "验证集比例应在 0-1 之间"
    assert config.patience > 0, "早停耐心值应为正数"
    assert config.epochs_pretrain > 0, "预训练轮数应为正数"
    assert 0 < config.fixed_test_ratio < 1.0, "测试集比例应在 0-1 之间"
    return True


# ============== 配置打印 ==============
def print_config(config: ExperimentConfig):
    """打印配置信息"""
    print("=" * 60)
    print("实验配置")
    print("=" * 60)
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.lr}")
    print(f"图像大小: {config.img_size}")
    print(f"工作线程数: {config.num_workers}")
    print(f"随机种子: {config.seed}")
    print(f"混合精度训练: {config.use_amp}")
    print(f"验证集比例: {config.val_split}")
    print(f"早停耐心值: {config.patience}")
    print(f"预训练轮数: {config.epochs_pretrain}")
    print(f"微调轮数列表: {config.epochs_finetune_list}")
    print(f"固定测试集比例: {config.fixed_test_ratio}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试配置
    config = get_default_config()
    print_config(config)
    validate_config(config)
    print("\n✓ 配置验证通过")
    
    print("\n数据集配置:")
    for name, path in DatasetConfig.DATASETS_MAP.items():
        print(f"  {name}: {path}") 