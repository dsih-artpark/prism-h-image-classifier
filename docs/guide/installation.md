# Installation Guide

This guide will help you set up the Prism-H mosquito breeding spot detection system on your local machine.

## Prerequisites

Before installing Prism-H, ensure you have the following prerequisites:

- **Python 3.11 or higher**
- **Git** (for cloning the repository)
- **Poetry** (recommended) or pip for dependency management
- **CUDA-compatible GPU** (optional, for faster training)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11+ | 3.11+ |
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | 50GB+ |
| GPU | None (CPU works) | CUDA 11.7+ |

## Installation Methods

### Method 1: Using Poetry (Recommended)

Poetry provides better dependency management and environment isolation.

#### 1. Install Poetry

=== "Linux/macOS"
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

=== "Windows"
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

#### 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd prism-h

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

### Method 2: Using pip and virtualenv

If you prefer using pip and virtual environments:

```bash
# Clone the repository
git clone <repository-url>
cd prism-h

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verification

Verify your installation by running:

```bash
# Check Python version
python --version

# Test GPU availability (if applicable)
python check_gpu.py

# Run a quick test
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Additional Dependencies

### For GPU Support

If you have a CUDA-compatible GPU, install the appropriate PyTorch version:

```bash
# For CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Apple Silicon (M1/M2) Macs

```bash
# Install with MPS (Metal Performance Shaders) support
pip install torch torchvision torchaudio
```

### For Development

If you plan to contribute to the project:

```bash
# Install development dependencies
poetry install --group dev

# Or with pip:
pip install -r requirements-dev.txt
```

## Configuration Files

After installation, you'll need to set up configuration files:

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your specific settings
nano .env
```

Example `.env` content:
```env
# Data paths
DATA_DIR=/path/to/your/data
RESULTS_DIR=/path/to/results

# HuggingFace token (if needed)
HF_TOKEN=your_huggingface_token_here

# GPU settings
CUDA_VISIBLE_DEVICES=0
```

### Model Configurations

The project uses several configuration files located in the `configs/` directory:

- `preprocessing.yaml` - Preprocessing parameters
- `simclr.yaml` - Self-supervised learning settings
- `classification.yaml` - Classification model parameters

## Troubleshooting

### Common Issues

#### Poetry Installation Problems

```bash
# If poetry command not found, add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall poetry
curl -sSL https://install.python-poetry.org | python3 - --uninstall
curl -sSL https://install.python-poetry.org | python3 -
```

#### CUDA/GPU Issues

```bash
# Check CUDA version
nvidia-smi

# Install specific PyTorch version
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Memory Issues

If you encounter out-of-memory errors:

```python
# Reduce batch size in configuration files
# Or set environment variable
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Permission Errors

```bash
# On Linux/macOS, you might need to adjust permissions
chmod +x scripts/*.py
```

### Platform-Specific Notes

=== "Linux"
    ```bash
    # Install system dependencies (Ubuntu/Debian)
    sudo apt-get update
    sudo apt-get install python3-dev build-essential
    ```

=== "macOS"
    ```bash
    # Install Xcode command line tools
    xcode-select --install
    
    # Install Homebrew (if not already installed)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

=== "Windows"
    ```powershell
    # Install Microsoft C++ Build Tools
    # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    ```

## Data Setup

After installation, you'll need to prepare your data:

1. **Image Data**: Place your images in the designated data directory
2. **Metadata**: Ensure JSON metadata files are properly formatted
3. **Directory Structure**: Follow the expected directory structure

```
data/
├── images/
│   ├── raw/
│   └── processed/
├── metadata/
│   └── annotations.json
└── results/
    ├── preprocessing/
    ├── embeddings/
    └── models/
```

## Next Steps

Once installation is complete:

1. [Quick Start](quickstart.md) - Run your first analysis
2. [Configuration](configuration.md) - Customize settings
3. [User Guide](overview.md) - Learn about all features

## Getting Help

If you encounter issues during installation:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the project's GitHub issues
3. Ensure all system requirements are met
4. Try the alternative installation method

!!! warning "Important"
    Make sure to activate your virtual environment (`poetry shell` or `source .venv/bin/activate`) before running any project commands.
