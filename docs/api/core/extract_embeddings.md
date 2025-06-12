# Feature Extraction API

The feature extraction module extracts meaningful visual representations from images using pre-trained SimCLR models. These embeddings capture semantic information about mosquito breeding spots and can be used for downstream tasks like clustering and classification.

## Overview

The feature extraction process:

1. **Model Loading**: Loads a pre-trained or fine-tuned SimCLR model
2. **Data Processing**: Applies standardized transforms to images
3. **Feature Extraction**: Generates dense feature vectors (embeddings)
4. **Storage**: Saves embeddings and metadata for further analysis

## Quick Start

```python
from prismh.core.extract_embeddings import extract_embeddings_main

# Extract embeddings using default configuration
extract_embeddings_main()
```

## Command Line Usage

```bash
# Basic feature extraction
python -m prismh.core.extract_embeddings \
    --input_dir /path/to/clean/images \
    --output_dir /path/to/embeddings

# With specific model and device
python -m prismh.core.extract_embeddings \
    --input_dir results/clean \
    --output_dir results/embeddings \
    --model_path models/simclr_finetuned.pt \
    --device cuda \
    --batch_size 64
```

## API Reference

::: prismh.core.extract_embeddings
    options:
      members:
        - extract_embeddings_main
        - SimCLRModel
        - PathBasedDataset
      show_root_heading: false
      show_source: true
      heading_level: 3

## Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | Auto-detect | Path to SimCLR checkpoint |
| `base_model` | `resnet50` | Backbone architecture |
| `output_dim` | 128 | Projection head output dimension |

### Processing Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Batch size for inference |
| `num_workers` | 0 | DataLoader worker processes |
| `device` | Auto-detect | Device (cpu/cuda/mps) |

### Data Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dir` | `preprocess_results/clean` | Clean images directory |
| `output_dir` | `simclr_finetuned` | Output directory |
| `image_size` | 224 | Input image size |

## Output Format

### Embeddings File

The extraction process generates `all_embeddings.npz` containing:

```python
# Load embeddings
data = np.load('all_embeddings.npz', allow_pickle=True)

embeddings = data['embeddings']      # Shape: (N, feature_dim)
file_paths = data['file_paths']      # Shape: (N,) - corresponding file paths
```

### File Structure

```
output_dir/
├── all_embeddings.npz              # Main embeddings file
├── train_embeddings.npz            # Training set embeddings
├── val_embeddings.npz              # Validation set embeddings
├── test_embeddings.npz             # Test set embeddings
└── extraction_metadata.json        # Extraction configuration
```

## Usage Examples

### Basic Extraction

```python
from prismh.core.extract_embeddings import extract_embeddings_main
from pathlib import Path
import numpy as np

def basic_extraction():
    # Run extraction with default settings
    extract_embeddings_main()
    
    # Load and examine results
    embeddings_file = Path("simclr_finetuned/all_embeddings.npz")
    if embeddings_file.exists():
        data = np.load(embeddings_file, allow_pickle=True)
        print(f"Extracted {len(data['embeddings'])} embeddings")
        print(f"Feature dimension: {data['embeddings'].shape[1]}")
    else:
        print("No embeddings found. Check configuration.")

basic_extraction()
```

### Custom Model Path

```python
from prismh.core.extract_embeddings import extract_embeddings_main
import os

def extract_with_custom_model():
    # Set custom model path
    os.environ['SIMCLR_MODEL_PATH'] = 'models/custom_simclr.pt'
    
    # Extract embeddings
    extract_embeddings_main()
    
    print("Extraction completed with custom model")

extract_with_custom_model()
```

### Batch Processing with Custom Configuration

```python
from prismh.core.extract_embeddings import SimCLRModel, PathBasedDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from pathlib import Path

def custom_extraction(image_dir, model_path, output_file, batch_size=32):
    """Custom feature extraction with full control"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimCLRModel(base_model='resnet50', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare data
    image_paths = list(Path(image_dir).glob("*.jpg"))
    dataset = PathBasedDataset(image_paths, transform=get_eval_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract embeddings
    all_embeddings = []
    all_paths = []
    
    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            if batch_images is not None:
                batch_images = batch_images.to(device)
                features, _ = model(batch_images)
                
                all_embeddings.append(features.cpu().numpy())
                all_paths.extend(batch_paths)
    
    # Save results
    embeddings = np.vstack(all_embeddings)
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        file_paths=np.array(all_paths)
    )
    
    print(f"Saved {len(embeddings)} embeddings to {output_file}")

# Usage
custom_extraction(
    image_dir="data/clean_images",
    model_path="models/simclr_best.pt",
    output_file="custom_embeddings.npz"
)
```

## Performance Optimization

### GPU Optimization

```python
import torch

# Optimize for GPU
if torch.cuda.is_available():
    # Enable memory efficiency
    torch.backends.cudnn.benchmark = True
    
    # Use larger batch sizes
    batch_size = 128
    
    # Enable pin memory
    pin_memory = True
else:
    batch_size = 32
    pin_memory = False
```

### Memory Management

```python
import gc
import torch

def memory_efficient_extraction():
    """Memory-efficient feature extraction"""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process in smaller batches
    batch_size = 32
    
    # Clear variables after use
    del model, embeddings
    gc.collect()
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_extraction(image_dirs, output_dir):
    """Extract embeddings from multiple directories in parallel"""
    
    def extract_single_dir(image_dir):
        dir_name = Path(image_dir).name
        output_file = Path(output_dir) / f"{dir_name}_embeddings.npz"
        
        # Run extraction for this directory
        custom_extraction(image_dir, "models/simclr.pt", output_file)
        return output_file
    
    # Process directories in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(extract_single_dir, dir_path) 
                  for dir_path in image_dirs]
        
        results = [future.result() for future in futures]
    
    print(f"Completed parallel extraction: {results}")
```

## Integration with Pipeline

### After Preprocessing

```python
from prismh.core.preprocess import ImagePreprocessor
from prismh.core.extract_embeddings import extract_embeddings_main

def preprocess_and_extract():
    """Complete preprocessing and feature extraction"""
    
    # Step 1: Preprocess images
    preprocessor = ImagePreprocessor(
        input_dir="raw_images",
        output_dir="processed"
    )
    preprocessor.run_preprocessing()
    
    # Step 2: Extract features from clean images
    # Update configuration to use clean images
    import os
    os.environ['CLEAN_IMAGES_DIR'] = 'processed/clean'
    
    extract_embeddings_main()
    
    print("Preprocessing and feature extraction completed")

preprocess_and_extract()
```

### Before Clustering

```python
from prismh.core.extract_embeddings import extract_embeddings_main
from prismh.core.cluster_embeddings import cluster_main

def extract_and_cluster():
    """Feature extraction followed by clustering"""
    
    # Extract embeddings
    extract_embeddings_main()
    
    # Run clustering on embeddings
    cluster_main()
    
    print("Feature extraction and clustering completed")

extract_and_cluster()
```

## Model Compatibility

### Supported Architectures

| Model | Backbone | Feature Dim | Use Case |
|-------|----------|-------------|----------|
| SimCLR-ResNet18 | ResNet-18 | 512 | Fast inference |
| SimCLR-ResNet50 | ResNet-50 | 2048 | Best performance |
| Custom SimCLR | Various | Configurable | Domain-specific |

### Loading Different Models

```python
# Load ImageNet pretrained
model = SimCLRModel(base_model='resnet50', pretrained=True)

# Load custom checkpoint
checkpoint = torch.load('custom_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Load fine-tuned model
model = SimCLRModel(base_model='resnet50', pretrained=False)
model.load_state_dict(torch.load('finetuned_simclr.pt'))
```

## Quality Assessment

### Embedding Quality Metrics

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

def assess_embedding_quality(embeddings_file):
    """Assess the quality of extracted embeddings"""
    
    data = np.load(embeddings_file)
    embeddings = data['embeddings']
    
    # Clustering-based quality assessment
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Silhouette score (higher is better)
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    # Embedding statistics
    mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
    std_norm = np.std(np.linalg.norm(embeddings, axis=1))
    
    metrics = {
        'silhouette_score': silhouette,
        'mean_embedding_norm': mean_norm,
        'std_embedding_norm': std_norm,
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1]
    }
    
    return metrics

# Assess quality
quality = assess_embedding_quality('all_embeddings.npz')
print(f"Embedding quality metrics: {quality}")
```

## Troubleshooting

### Common Issues

**Model not found:**
```python
# Check model path and ensure it exists
model_path = Path("models/simclr_model.pt")
if not model_path.exists():
    print(f"Model not found at {model_path}")
    print("Train a SimCLR model first or download a pretrained one")
```

**Out of memory:**
```python
# Reduce batch size
batch_size = 16  # Instead of 64

# Clear GPU cache
torch.cuda.empty_cache()

# Use CPU if necessary
device = torch.device('cpu')
```

**Inconsistent image sizes:**
```python
# Ensure all images are properly resized
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## Advanced Usage

### Custom Feature Extractors

```python
class CustomFeatureExtractor:
    def __init__(self, model_path, device='auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
    
    def extract_features(self, image_paths, batch_size=32):
        """Extract features from a list of image paths"""
        dataset = PathBasedDataset(image_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        features = []
        with torch.no_grad():
            for batch in dataloader:
                batch_features = self.model.encoder(batch.to(self.device))
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)

# Usage
extractor = CustomFeatureExtractor('models/custom_simclr.pt')
features = extractor.extract_features(image_paths)
```

## Related Documentation

- [SimCLR Training](../models/simclr.md) - Train custom feature extractors
- [Clustering Analysis](cluster_embeddings.md) - Use embeddings for clustering
- [Classification](../models/classify.md) - Downstream classification tasks
- [Preprocessing](preprocess.md) - Prepare images for feature extraction 