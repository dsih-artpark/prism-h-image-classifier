# Configuration Guide

This guide explains how to configure Prism-H for different use cases and environments. The system provides multiple configuration methods to suit different workflows.

## Configuration Methods

### 1. Command Line Arguments

The most common way to configure the system:

```bash
# Preprocessing configuration
python -m prismh.core.preprocess \
    --data_dir /path/to/images \
    --output_dir results \
    --ccthreshold 0.9 \
    --outlier_distance 0.68 \
    --sample_size 5000

# Feature extraction configuration
python -m prismh.core.extract_embeddings \
    --input_dir results/clean \
    --output_dir results/embeddings \
    --batch_size 64 \
    --device cuda
```

### 2. Environment Variables

Set system-wide defaults:

```bash
# Data paths
export PRISMH_DATA_DIR="/path/to/default/data"
export PRISMH_OUTPUT_DIR="/path/to/default/results"

# Model configuration
export PRISMH_MODEL_PATH="/path/to/simclr/model.pt"
export PRISMH_DEVICE="cuda"

# Processing parameters
export PRISMH_BATCH_SIZE="64"
export PRISMH_NUM_WORKERS="4"
```

### 3. Configuration Files

Create YAML configuration files for complex setups:

```yaml
# config/preprocessing.yaml
preprocessing:
  ccthreshold: 0.9
  outlier_distance: 0.68
  sample_size: 10000
  quality_thresholds:
    dark_threshold: 13
    blur_threshold: 50

# config/simclr.yaml
simclr:
  model:
    base_model: "resnet50"
    output_dim: 128
    pretrained: true
  training:
    batch_size: 32
    learning_rate: 0.001
    epochs: 100
    temperature: 0.5

# config/extraction.yaml
extraction:
  batch_size: 64
  num_workers: 4
  device: "auto"
  image_size: 224
```

### 4. Python Configuration

Direct configuration in Python code:

```python
from prismh.core.preprocess import ImagePreprocessor
from prismh.config import Config

# Using configuration objects
config = Config({
    'preprocessing': {
        'ccthreshold': 0.85,
        'outlier_distance': 0.70
    },
    'extraction': {
        'batch_size': 32,
        'device': 'cuda'
    }
})

# Initialize with configuration
preprocessor = ImagePreprocessor(
    input_dir="data/images",
    output_dir="results",
    **config.preprocessing
)
```

## Module-Specific Configuration

### Preprocessing Configuration

#### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `ccthreshold` | float | 0.9 | 0.0-1.0 | Similarity threshold for duplicate detection |
| `outlier_distance` | float | 0.68 | 0.0-1.0 | Distance threshold for outlier detection |
| `sample_size` | int | None | >0 | Number of images to process (None for all) |

#### Quality Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dark_threshold` | int | 13 | Mean brightness threshold for dark images |
| `blur_threshold` | int | 50 | Variance threshold for blur detection |
| `min_file_size` | int | 1024 | Minimum file size in bytes |

#### Example Configuration

```python
# Conservative settings (higher quality)
conservative_config = {
    'ccthreshold': 0.95,        # Very strict duplicate detection
    'outlier_distance': 0.60,   # More aggressive outlier removal
    'dark_threshold': 20,       # Higher brightness requirement
    'blur_threshold': 100       # Stricter blur detection
}

# Permissive settings (keep more images)
permissive_config = {
    'ccthreshold': 0.80,        # More lenient duplicate detection
    'outlier_distance': 0.80,   # Keep more outliers
    'dark_threshold': 8,        # Accept darker images
    'blur_threshold': 30        # Accept more blur
}
```

### SimCLR Training Configuration

#### Model Architecture

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `base_model` | str | "resnet50" | resnet18, resnet50 | Backbone architecture |
| `output_dim` | int | 128 | 64, 128, 256 | Projection head output dimension |
| `pretrained` | bool | true | true, false | Use ImageNet pretrained weights |

#### Training Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `batch_size` | int | 32 | 8-512 | Training batch size |
| `learning_rate` | float | 0.001 | 1e-5 to 1e-2 | Initial learning rate |
| `epochs` | int | 100 | 1-1000 | Number of training epochs |
| `temperature` | float | 0.5 | 0.1-1.0 | Contrastive loss temperature |
| `weight_decay` | float | 1e-4 | 0-1e-2 | L2 regularization strength |

#### Data Augmentation

```python
# Default augmentation configuration
augmentation_config = {
    'resize': 256,
    'crop_size': 224,
    'horizontal_flip_prob': 0.5,
    'color_jitter': {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.4,
        'hue': 0.1,
        'prob': 0.8
    },
    'grayscale_prob': 0.2,
    'gaussian_blur': {
        'kernel_size': 23,
        'sigma': [0.1, 2.0],
        'prob': 0.5
    }
}

# Strong augmentation for challenging datasets
strong_augmentation = {
    'color_jitter': {
        'brightness': 0.8,
        'contrast': 0.8,
        'saturation': 0.8,
        'hue': 0.2,
        'prob': 0.8
    },
    'gaussian_blur': {
        'prob': 0.8
    }
}
```

### Feature Extraction Configuration

#### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 64 | Inference batch size |
| `num_workers` | int | 0 | DataLoader worker processes |
| `device` | str | "auto" | Device (cpu/cuda/mps/auto) |
| `pin_memory` | bool | true | Enable memory pinning for GPU |

#### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | auto | Path to trained SimCLR model |
| `checkpoint_key` | str | "model_state_dict" | Key for model weights in checkpoint |
| `strict_loading` | bool | true | Strict state dict loading |

### Clustering Configuration

#### Fastdup Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.9 | Similarity threshold for clustering |
| `min_cluster_size` | int | 2 | Minimum images per cluster |
| `ccthreshold` | float | 0.96 | Connected components threshold |

#### Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_images_per_cluster` | int | 50 | Maximum images shown per cluster |
| `image_size` | tuple | (224, 224) | Display image size |
| `gallery_format` | str | "html" | Output format (html/json) |

## Environment-Specific Configuration

### Development Environment

```yaml
# config/dev.yaml
development:
  preprocessing:
    sample_size: 1000      # Small sample for fast iteration
    ccthreshold: 0.85      # Moderate quality filtering
  
  simclr:
    training:
      epochs: 10           # Quick training
      batch_size: 16       # Small batch for limited GPU memory
  
  extraction:
    batch_size: 32         # Conservative batch size
    device: "cpu"          # Fallback to CPU if needed
```

### Production Environment

```yaml
# config/prod.yaml
production:
  preprocessing:
    sample_size: null      # Process all images
    ccthreshold: 0.92      # High quality filtering
  
  simclr:
    training:
      epochs: 200          # Thorough training
      batch_size: 64       # Utilize full GPU capacity
  
  extraction:
    batch_size: 128        # Large batch for efficiency
    device: "cuda"         # GPU acceleration
    num_workers: 8         # Parallel data loading
```

### Cloud/HPC Environment

```yaml
# config/cloud.yaml
cloud:
  preprocessing:
    sample_size: 100000    # Large-scale processing
  
  simclr:
    training:
      batch_size: 256      # Large batch for distributed training
      num_gpus: 4          # Multi-GPU setup
  
  extraction:
    batch_size: 512        # High-throughput processing
    distributed: true      # Distributed processing
```

## Hardware-Specific Optimization

### GPU Configuration

```python
# NVIDIA GPU optimization
gpu_config = {
    'device': 'cuda',
    'batch_size': 128,
    'num_workers': 8,
    'pin_memory': True,
    'mixed_precision': True,
    'compile_model': True  # PyTorch 2.0+
}

# Multi-GPU configuration
multi_gpu_config = {
    'device': 'cuda',
    'data_parallel': True,
    'devices': [0, 1, 2, 3],
    'batch_size': 256,  # Total batch size across GPUs
    'sync_batchnorm': True
}
```

### CPU Configuration

```python
# CPU optimization
cpu_config = {
    'device': 'cpu',
    'batch_size': 32,
    'num_workers': 4,  # Number of CPU cores
    'pin_memory': False,
    'mixed_precision': False
}
```

### Apple Silicon (M1/M2) Configuration

```python
# Apple Silicon optimization
mps_config = {
    'device': 'mps',
    'batch_size': 64,
    'num_workers': 0,  # MPS works best with num_workers=0
    'pin_memory': False
}
```

## Dataset-Specific Configuration

### Large Dataset (>100k images)

```yaml
large_dataset:
  preprocessing:
    sample_size: null
    ccthreshold: 0.90
    outlier_distance: 0.65
    
  extraction:
    batch_size: 128
    streaming: true        # Stream data to reduce memory usage
    checkpoint_frequency: 1000
    
  clustering:
    max_samples: 50000     # Subsample for clustering if needed
    threshold: 0.92
```

### Small Dataset (<10k images)

```yaml
small_dataset:
  preprocessing:
    ccthreshold: 0.85      # More permissive to retain data
    outlier_distance: 0.75
    
  simclr:
    training:
      epochs: 300          # More epochs for small datasets
      batch_size: 16       # Smaller batches
      augmentation_strength: 'strong'
      
  clustering:
    min_cluster_size: 1    # Allow singleton clusters
```

### Noisy Dataset

```yaml
noisy_dataset:
  preprocessing:
    ccthreshold: 0.95      # Strict duplicate removal
    outlier_distance: 0.60 # Aggressive outlier removal
    dark_threshold: 25     # Higher quality requirements
    blur_threshold: 80
    
  simclr:
    training:
      temperature: 0.3     # Lower temperature for noisy data
      weight_decay: 1e-3   # More regularization
```

## Advanced Configuration

### Custom Configuration Classes

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PreprocessingConfig:
    ccthreshold: float = 0.9
    outlier_distance: float = 0.68
    sample_size: Optional[int] = None
    dark_threshold: int = 13
    blur_threshold: int = 50
    
    def validate(self):
        assert 0.0 <= self.ccthreshold <= 1.0
        assert 0.0 <= self.outlier_distance <= 1.0
        if self.sample_size is not None:
            assert self.sample_size > 0

@dataclass
class SimCLRConfig:
    base_model: str = "resnet50"
    output_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 0.001
    temperature: float = 0.5
    epochs: int = 100
    
    def __post_init__(self):
        assert self.base_model in ["resnet18", "resnet50"]
        assert self.output_dim in [64, 128, 256]

# Usage
config = PreprocessingConfig(ccthreshold=0.85, sample_size=5000)
config.validate()
```

### Configuration Inheritance

```python
class BaseConfig:
    def __init__(self):
        self.load_defaults()
    
    def load_defaults(self):
        self.preprocessing = PreprocessingConfig()
        self.simclr = SimCLRConfig()
    
    def update_from_file(self, config_file):
        import yaml
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        self._update_from_dict(config_data)
    
    def update_from_env(self):
        import os
        if 'PRISMH_CCTHRESHOLD' in os.environ:
            self.preprocessing.ccthreshold = float(os.environ['PRISMH_CCTHRESHOLD'])
        # ... more environment variables

class DevelopmentConfig(BaseConfig):
    def load_defaults(self):
        super().load_defaults()
        self.preprocessing.sample_size = 1000
        self.simclr.epochs = 10

class ProductionConfig(BaseConfig):
    def load_defaults(self):
        super().load_defaults()
        self.preprocessing.ccthreshold = 0.92
        self.simclr.epochs = 200
```

### Dynamic Configuration

```python
def get_config_for_dataset(dataset_path):
    """Automatically configure based on dataset characteristics"""
    import os
    
    # Count images
    image_count = len([f for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if image_count < 5000:
        return SmallDatasetConfig()
    elif image_count > 100000:
        return LargeDatasetConfig()
    else:
        return StandardConfig()

def auto_configure_hardware():
    """Automatically configure based on available hardware"""
    import torch
    
    config = {}
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory > 16e9:  # 16GB+
            config['batch_size'] = 128
        elif gpu_memory > 8e9:  # 8GB+
            config['batch_size'] = 64
        else:
            config['batch_size'] = 32
        config['device'] = 'cuda'
    else:
        config['batch_size'] = 16
        config['device'] = 'cpu'
    
    return config
```

## Configuration Validation

### Parameter Validation

```python
def validate_preprocessing_config(config):
    """Validate preprocessing configuration"""
    errors = []
    
    if not 0.0 <= config.ccthreshold <= 1.0:
        errors.append("ccthreshold must be between 0.0 and 1.0")
    
    if not 0.0 <= config.outlier_distance <= 1.0:
        errors.append("outlier_distance must be between 0.0 and 1.0")
    
    if config.sample_size is not None and config.sample_size <= 0:
        errors.append("sample_size must be positive")
    
    if errors:
        raise ValueError("Configuration errors: " + "; ".join(errors))

def validate_hardware_config(config):
    """Validate hardware configuration"""
    import torch
    
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    
    if config.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but not available")
    
    if config.batch_size > 512:
        print("Warning: Very large batch size may cause memory issues")
```

### Configuration Compatibility

```python
def check_config_compatibility(preprocess_config, simclr_config):
    """Check compatibility between different module configurations"""
    warnings = []
    
    # Check if sample size is appropriate for training epochs
    if (preprocess_config.sample_size and 
        preprocess_config.sample_size < 1000 and 
        simclr_config.epochs > 50):
        warnings.append("Small sample size with many epochs may cause overfitting")
    
    # Check batch size vs dataset size
    if (preprocess_config.sample_size and 
        simclr_config.batch_size > preprocess_config.sample_size // 10):
        warnings.append("Batch size may be too large for dataset size")
    
    for warning in warnings:
        print(f"Warning: {warning}")
```

## Best Practices

### Configuration Management

1. **Use version control** for configuration files
2. **Separate configs by environment** (dev/staging/prod)
3. **Document configuration changes** and their impact
4. **Validate configurations** before running experiments
5. **Use meaningful parameter names** and comments

### Performance Optimization

1. **Profile different configurations** to find optimal settings
2. **Monitor resource usage** (GPU memory, CPU, disk I/O)
3. **Adjust batch sizes** based on available hardware
4. **Use mixed precision** for compatible hardware
5. **Enable compilation** with PyTorch 2.0+

### Reproducibility

```python
# Ensure reproducible results
reproducibility_config = {
    'random_seed': 42,
    'deterministic': True,
    'benchmark': False,  # Disable cudnn benchmark for reproducibility
    'num_workers': 0     # Avoid multiprocessing for deterministic results
}

# Set seeds
import torch
import numpy as np
import random

def set_reproducible_config(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Configuration Examples by Use Case

### Research/Experimentation

```yaml
research:
  preprocessing:
    sample_size: 5000
    ccthreshold: 0.85
  simclr:
    epochs: 50
    batch_size: 32
    save_frequency: 10
  logging:
    level: DEBUG
    tensorboard: true
    save_embeddings: true
```

### Production Deployment

```yaml
production:
  preprocessing:
    ccthreshold: 0.92
    quality_checks: strict
  extraction:
    batch_size: 128
    optimization: maximum
  monitoring:
    metrics: true
    alerts: true
    performance_tracking: true
```

### Edge/Mobile Deployment

```yaml
edge:
  model:
    quantization: int8
    pruning: 0.3
  processing:
    batch_size: 1
    memory_limit: 512MB
  optimization:
    model_compression: true
    inference_only: true
```

This configuration system provides the flexibility to adapt Prism-H to various environments, datasets, and use cases while maintaining reproducibility and performance.
