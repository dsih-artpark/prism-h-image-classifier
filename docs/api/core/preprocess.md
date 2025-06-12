# Image Preprocessing API

The preprocessing module provides comprehensive image quality assessment and data cleaning capabilities for mosquito breeding spot images.

## Overview

The `ImagePreprocessor` class uses the fastdup library to automatically identify and categorize problematic images, including:

- **Invalid images**: Corrupted or unreadable files
- **Duplicates**: Nearly identical or repeated images
- **Outliers**: Images significantly different from the dataset
- **Dark images**: Poorly lit or underexposed images
- **Blurry images**: Out-of-focus or motion-blurred images

## Quick Start

```python
from prismh.core.preprocess import ImagePreprocessor

# Initialize the preprocessor
preprocessor = ImagePreprocessor(
    input_dir="path/to/raw/images",
    output_dir="path/to/results",
    ccthreshold=0.9,
    outlier_distance=0.68
)

# Run the complete preprocessing pipeline
preprocessor.run_preprocessing()
```

## Command Line Usage

```bash
# Basic usage
python -m prismh.core.preprocess --data_dir images/ --output_dir results/

# With custom parameters
python -m prismh.core.preprocess \
    --data_dir /path/to/images \
    --output_dir /path/to/results \
    --ccthreshold 0.85 \
    --outlier_distance 0.70
```

## API Reference

::: prismh.core.preprocess
    options:
      members:
        - ImagePreprocessor
      show_root_heading: false
      show_source: true
      heading_level: 3

## Configuration Parameters

### Quality Thresholds

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `ccthreshold` | Similarity threshold for duplicate detection | 0.9 | 0.0-1.0 |
| `outlier_distance` | Distance threshold for outlier detection | 0.68 | 0.0-1.0 |

### Image Quality Metrics

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Mean brightness | < 13 | Detect dark images |
| Blur variance | < 50 | Detect blurry images |
| File validity | - | Detect corrupted files |

## Output Structure

The preprocessing pipeline creates the following directory structure:

```
output_dir/
├── clean/                    # High-quality images
└── problematic/             # Filtered images
    ├── invalid/             # Corrupted files
    ├── duplicates/          # Duplicate images
    ├── outliers/            # Unusual images
    ├── dark/                # Dark/underexposed images
    └── blurry/              # Blurry images
```

## Usage Examples

### Basic Preprocessing

```python
from prismh.core.preprocess import ImagePreprocessor

# Simple preprocessing
preprocessor = ImagePreprocessor(
    input_dir="raw_images/",
    output_dir="processed/"
)
preprocessor.run_preprocessing()
```

### Custom Configuration

```python
# Advanced configuration
preprocessor = ImagePreprocessor(
    input_dir="raw_images/",
    output_dir="processed/",
    ccthreshold=0.85,        # More strict duplicate detection
    outlier_distance=0.75    # More lenient outlier detection
)
preprocessor.run_preprocessing()
```

### With Metadata Integration

```python
import json
from pathlib import Path

# Load metadata
metadata_path = Path("metadata/annotations.json")
with open(metadata_path) as f:
    metadata = json.load(f)

# Process with metadata context
preprocessor = ImagePreprocessor(
    input_dir="raw_images/",
    output_dir="processed/"
)
preprocessor.run_preprocessing()

# Analyze results with metadata
results_summary = {
    "total_processed": len(list(Path("processed/clean").glob("*"))),
    "problematic_count": len(list(Path("processed/problematic").rglob("*"))),
    "metadata_entries": len(metadata)
}
```

## Performance Optimization

### Memory Management

```python
# For large datasets, process in batches
import os
os.environ['FASTDUP_BATCH_SIZE'] = '1000'

preprocessor = ImagePreprocessor(
    input_dir="large_dataset/",
    output_dir="results/"
)
```

### Parallel Processing

```python
# Utilize multiple CPU cores
import multiprocessing
os.environ['FASTDUP_NUM_WORKERS'] = str(multiprocessing.cpu_count())
```

## Quality Metrics

The preprocessing pipeline provides detailed quality metrics:

```python
# Access quality statistics
stats = preprocessor.get_quality_stats()
print(f"Quality score: {stats['quality_score']:.2f}")
print(f"Duplicate rate: {stats['duplicate_rate']:.1%}")
print(f"Outlier rate: {stats['outlier_rate']:.1%}")
```

## Integration with Other Modules

### With Feature Extraction

```python
from prismh.core.preprocess import ImagePreprocessor
from prismh.core.extract_embeddings import extract_embeddings_main

# Step 1: Preprocess
preprocessor = ImagePreprocessor("raw/", "processed/")
preprocessor.run_preprocessing()

# Step 2: Extract features from clean images
# (Configure paths in extract_embeddings.py)
extract_embeddings_main()
```

### With Clustering

```python
from prismh.core.cluster_embeddings import cluster_main

# After preprocessing and feature extraction
cluster_main()
```

## Troubleshooting

### Common Issues

**Memory errors with large datasets:**
```python
# Reduce batch size or process in chunks
os.environ['FASTDUP_BATCH_SIZE'] = '500'
```

**Path-related errors:**
```python
# Use absolute paths
from pathlib import Path
input_path = Path("images").resolve()
output_path = Path("results").resolve()
```

**Permission errors:**
```bash
# Ensure write permissions
chmod -R 755 output_directory/
```

## Advanced Features

### Custom Quality Filters

```python
class CustomImagePreprocessor(ImagePreprocessor):
    def custom_quality_check(self, image_path):
        """Add custom quality assessment logic"""
        # Implement custom quality checks
        pass
    
    def run_preprocessing(self):
        # Run standard preprocessing
        super().run_preprocessing()
        # Add custom processing steps
        self.apply_custom_filters()
```

### Batch Processing

```python
def process_multiple_datasets(datasets):
    """Process multiple image datasets"""
    for dataset_info in datasets:
        preprocessor = ImagePreprocessor(
            input_dir=dataset_info['input'],
            output_dir=dataset_info['output']
        )
        preprocessor.run_preprocessing()
        print(f"Completed: {dataset_info['name']}")
```

## Related Documentation

- [Feature Extraction](extract_embeddings.md) - Next step in the pipeline
- [Clustering Analysis](cluster_embeddings.md) - Pattern discovery
- [Configuration Guide](../../guide/configuration.md) - Detailed parameter tuning
- [Examples](../../examples/basic_usage.md) - Practical usage scenarios 