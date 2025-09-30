# config.py
# -*- coding: utf-8 -*-
"""
Experiment configuration file
Contains all experiment-related configuration parameters, dataset paths and constant definitions
"""
from dataclasses import dataclass
from typing import Dict


# ============== Experiment Configuration Classes ==============
@dataclass
class ExperimentConfig:
    """Experiment configuration class"""
    # Training hyperparameters
    batch_size: int = 32
    lr: float = 1e-3
    img_size: int = 224
    num_workers: int = 4
    seed: int = 42
    
    # Training strategy
    use_amp: bool = True                # Whether to use mixed precision training
    val_split: float = 0.2              # Pretraining validation split ratio
    patience: int = 5                   # Early stopping patience value
    
    # Training epochs
    epochs_pretrain: int = 50           # Pretraining epochs
    epochs_finetune_list: list = None   # Fine-tuning epochs list, e.g. [10, 20, 30]
    
    # Data split
    fixed_test_ratio: float = 0.1       # Fixed test set ratio (10%)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.epochs_finetune_list is None:
            self.epochs_finetune_list = [10, 20, 30]


# ============== Dataset Configuration ==============
class DatasetConfig:
    """Dataset path configuration"""
    
    # Office-31 dataset path
    BASE_PATH = '/home/exp/exp/ruih/newdr/Office-31-20250831T042022Z-1-001/Office-31/Data'
    
    DATASETS_MAP = {
        "amazon": f'{BASE_PATH}/amazon',
        "dslr": f'{BASE_PATH}/dslr',
        "webcam": f'{BASE_PATH}/webcam',
    }
    
    # Available dataset names
    AVAILABLE_DATASETS = list(DATASETS_MAP.keys())
    
    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """Get the path of specified dataset"""
        if dataset_name not in cls.DATASETS_MAP:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {cls.AVAILABLE_DATASETS}")
        return cls.DATASETS_MAP[dataset_name]
    
    @classmethod
    def validate_datasets(cls, dataset_names: list) -> bool:
        """Validate whether all dataset names are valid"""
        for name in dataset_names:
            if name not in cls.DATASETS_MAP:
                raise ValueError(f"Unknown dataset: {name}. Available datasets: {cls.AVAILABLE_DATASETS}")
        return True


# ============== Model Configuration ==============
class ModelConfig:
    """Model related configuration"""
    
    # Supported initialization modes
    INIT_MODES = [
        "resnet50_imagenet",      # ResNet50 + ImageNet pretrained weights
        "random",                  # Completely random initialization
        "random_then_fixed_src",   # Random initialization + fixed source dataset pretraining
    ]
    
    # Model architecture
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 31  # Office-31 dataset has 31 classes
    
    @classmethod
    def validate_init_mode(cls, init_mode: str) -> bool:
        """Validate whether initialization mode is valid"""
        if init_mode not in cls.INIT_MODES:
            raise ValueError(f"Unknown initialization mode: {init_mode}. Available modes: {cls.INIT_MODES}")
        return True


# ============== Image Processing Constants ==============
class ImageConfig:
    """Image preprocessing related configuration"""
    
    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Default image size
    DEFAULT_IMG_SIZE = 224
    


# ============== Experiment Output Configuration ==============
class OutputConfig:
    """Experiment output related configuration"""
    
    # Default output directory
    DEFAULT_OUTPUT_DIR = "./exp_out"
    
    # Result filename format
    RESULT_FILENAME_FORMAT = "{source}_to_{target}_{init_mode}_r{split}_e{epoch}.json"
    
    # Summary filename
    SUMMARY_FILENAME = "summary.csv"
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'


# ============== CLI Argument Defaults ==============
class CLIDefaults:
    """Default values for command line arguments"""
    
    # Fixed source dataset
    FIXED_SOURCE = "amazon"
    
    # Training set usage ratio (relative to training candidate set)
    SPLITS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    # Initialization modes
    INIT_MODES = ["resnet50_imagenet", "random", "random_then_fixed_src"]
    
    # Fine-tuning epochs
    EPOCHS_FINETUNE = [100]
    
    # Pretraining epochs
    EPOCHS_PRETRAIN = 50
    
    # Validation split ratio
    VAL_SPLIT = 0.2
    
    # Early stopping patience
    PATIENCE = 5
    
    # Training hyperparameters
    BATCH_SIZE = 32
    LR = 1e-3
    IMG_SIZE = 224
    NUM_WORKERS = 4
    SEED = 42
    
    # Target mode
    TARGET_MODE = "merged"  # "merged" or "separate"


# ============== Global Configuration Instance ==============
# Create default configuration instance
def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration"""
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


# ============== Configuration Validation ==============
def validate_config(config: ExperimentConfig) -> bool:
    """Validate whether configuration is reasonable"""
    assert 0 < config.batch_size <= 256, "batch_size should be between 1-256"
    assert 0 < config.lr <= 1.0, "learning rate should be between 0-1"
    assert config.img_size > 0, "image size should be positive"
    assert 0 <= config.val_split < 1.0, "validation split should be between 0-1"
    assert config.patience > 0, "patience should be positive"
    assert config.epochs_pretrain > 0, "pretraining epochs should be positive"
    assert 0 < config.fixed_test_ratio < 1.0, "test set ratio should be between 0-1"
    return True


# ============== Configuration Printing ==============
def print_config(config: ExperimentConfig):
    """Print configuration information"""
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Image size: {config.img_size}")
    print(f"Number of workers: {config.num_workers}")
    print(f"Random seed: {config.seed}")
    print(f"Mixed precision training: {config.use_amp}")
    print(f"Validation split: {config.val_split}")
    print(f"Early stopping patience: {config.patience}")
    print(f"Pretraining epochs: {config.epochs_pretrain}")
    print(f"Fine-tuning epochs list: {config.epochs_finetune_list}")
    print(f"Fixed test set ratio: {config.fixed_test_ratio}")
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print_config(config)
    validate_config(config)
    print("\n✓ Configuration validation passed")
    
    print("\nDataset configuration:")
    for name, path in DatasetConfig.DATASETS_MAP.items():
        print(f"  {name}: {path}") 