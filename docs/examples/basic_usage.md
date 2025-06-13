# Basic Usage Examples

This page provides practical examples of using Prism-H for mosquito breeding spot detection and analysis.

## Complete Workflow Example

Here's a complete example that demonstrates the entire pipeline:

```python
#!/usr/bin/env python3
"""Complete Prism-H workflow example"""

from pathlib import Path
from prismh.core.preprocess import ImagePreprocessor

def main():
    # Configuration
    config = {
        'data_dir': '/Users/kirubeso.r/Documents/ArtPark/all_them_images',
        'metadata_file': '/Users/kirubeso.r/Documents/ArtPark/all_jsons/first100k.json',
        'output_dir': 'results_complete_workflow',
        'sample_size': 5000
    }
    
    print("🔍 Starting Prism-H Complete Workflow")
    
    # Step 1: Data Preprocessing
    print("\n📋 Step 1: Data Preprocessing")
    preprocessor = ImagePreprocessor(
        input_dir=config['data_dir'],
        output_dir=config['output_dir'],
        ccthreshold=0.9,
        outlier_distance=0.68
    )
    preprocessor.run_preprocessing()
    
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()
```

## Command Line Examples

### Basic Usage

```bash
# Preprocess images
python -m prismh.core.preprocess \
    --data_dir /path/to/images \
    --output_dir results \
    --sample_size 5000

# Extract features
python -m prismh.core.extract_embeddings \
    --input_dir results/clean \
    --output_dir results/embeddings

# Run clustering
python -m prismh.core.cluster_embeddings \
    --embeddings results/embeddings/all_embeddings.npz \
    --output_dir results/clustering
```

## Step-by-Step Examples

### 1. Basic Preprocessing

```python
from prismh.core.preprocess import ImagePreprocessor

def basic_preprocessing():
    preprocessor = ImagePreprocessor(
        input_dir="data/raw_images",
        output_dir="results/preprocessing"
    )
    
    # Run preprocessing
    preprocessor.run_preprocessing()
    
    # Check results
    clean_dir = Path("results/preprocessing/clean")
    print(f"Clean images: {len(list(clean_dir.glob('*')))}")

basic_preprocessing()
```

### 2. Feature Extraction

```python
from prismh.core.extract_embeddings import extract_embeddings_main

def extract_features():
    # Extract embeddings from clean images
    extract_embeddings_main()
    
    print("Feature extraction completed")

extract_features()
```

### 3. Clustering Analysis

```python
from prismh.core.cluster_embeddings import cluster_main

def run_clustering():
    # Run clustering on extracted embeddings
    cluster_main()
    
    print("Clustering analysis completed")

run_clustering()
```

## Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name):
    """Monitor operation performance"""
    start_time = time.time()
    print(f"🚀 Starting {operation_name}")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"{operation_name} completed in {duration:.2f} seconds")

# Usage example
with performance_monitor("Image Preprocessing"):
    preprocessor = ImagePreprocessor("data/images", "results")
    preprocessor.run_preprocessing()
```

## Integration Examples

### Batch Processing

```python
def process_multiple_datasets(datasets):
    """Process multiple image datasets"""
    for dataset_info in datasets:
        print(f"Processing {dataset_info['name']}...")
        
        preprocessor = ImagePreprocessor(
            input_dir=dataset_info['input'],
            output_dir=dataset_info['output']
        )
        preprocessor.run_preprocessing()
        
        print(f"Completed: {dataset_info['name']}")

# Example usage
datasets = [
    {'name': 'Dataset_A', 'input': 'data/a', 'output': 'results/a'},
    {'name': 'Dataset_B', 'input': 'data/b', 'output': 'results/b'}
]

process_multiple_datasets(datasets)
```

## Common Use Cases

### Processing Sample Data

```python
# Process a small sample for testing
def process_sample():
    preprocessor = ImagePreprocessor(
        input_dir="data/sample",
        output_dir="results/sample",
        ccthreshold=0.9,
        outlier_distance=0.68
    )
    
    preprocessor.run_preprocessing()
    
    # Print summary
    clean_count = len(list(Path("results/sample/clean").glob("*")))
    print(f"Processed sample: {clean_count} clean images")

process_sample()
```

### Custom Configuration

```python
# Advanced preprocessing with custom settings
def advanced_preprocessing():
    preprocessor = ImagePreprocessor(
        input_dir="data/large_dataset",
        output_dir="results/advanced",
        ccthreshold=0.85,        # More strict duplicate detection
        outlier_distance=0.75    # More lenient outlier detection
    )
    
    preprocessor.run_preprocessing()

advanced_preprocessing()
```

## Next Steps

After completing these examples:

- [Configuration Guide](../guide/configuration.md) - Customize parameters
- [API Reference](../api/core/preprocess.md) - Detailed function documentation
- [Advanced Examples](advanced.md) - Complex workflows 