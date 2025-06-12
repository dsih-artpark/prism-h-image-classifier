# Prism-H - Mosquito Breeding Spot Analysis

## Description

Aiming to support ASHA workers, this project employs AI models to analyze images of potential mosquito breeding spots, reducing data errors and improving the targeting of intervention efforts. It includes components for image preprocessing, feature extraction (embeddings), clustering, classification, object detection, and metadata integration.

## Installation

### Prerequisites
- Python 3.11 or higher
- Poetry (for dependency management)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd prism-h-image-classifier
   ```

2. **Install with Poetry:**
   ```bash
   poetry install
   ```

3. **Activate the environment:**
   ```bash
   poetry shell
   ```

### Alternative Setup (if you don't have Poetry)

1. **Install Poetry first:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Then follow the Quick Setup steps above**

## Usage

After installation, you can run the various scripts in the project:

```bash
# Example usage
python classify.py
python preprocess.py
```

## Project Structure

This repository contains the following key components:

*   **Core Processing Scripts:**
    *   `preprocess.py`: Uses fastdup library to identify and categorize problematic images (invalid, duplicates, outliers, dark, blurry) and separates them from clean images.
    *   `json_processor.py`: Processes JSON datasets containing image URLs - counts image extensions, analyzes invalid URLs, downloads images, and can display random samples.
    *   `extract_embeddings.py`: Extracts feature embeddings from images using a pretrained/finetuned SimCLR model and saves them as compressed NPZ files.
    *   `cluster_embeddings.py`: Uses fastdup to cluster image embeddings and generates visualization galleries showing clusters, outliers, and potential duplicates.
    *   `classify.py`: Trains an image classifier on top of a SimCLR encoder that categorizes images into different classes, with support for model validation and testing.
    *   `metadata_integrator.py`: Integrates image data with JSON metadata, performs analysis, generates visualizations/reports, and can create annotated images with metadata overlays.
*   **AI Models & Notebooks:**
    *   `simclr.py`: Implements a self-supervised learning model (SimCLR) for creating image feature representations without requiring labels.
    *   `object_detection.ipynb`: A Jupyter Notebook exploring or implementing object detection using a stack involving Grounding-DINO and Owl-ViT models.
*   **Data:**
    *   `jsons/`: Directory containing JSON dataset files with image metadata.
*   **Configuration:**
    *   `pyproject.toml`, `poetry.lock`: Poetry files for managing Python dependencies. (Generated during setup)
    *   `.gitignore`: Specifies intentionally untracked files for Git.

## Workflow Overview

1.  **Data Input:** Image data (location specified via configuration or arguments) and metadata from the `jsons/` directory are used.
2.  **Preprocessing:** Images are processed by `preprocess.py` to filter out problematic images, while `json_processor.py` handles downloading and organizing images from metadata.
3.  **Feature Extraction:** Embeddings are generated using models defined/used in `simclr.py` via the `extract_embeddings.py` script.
4.  **Analysis:**
    *   Image embeddings are clustered using `cluster_embeddings.py` to identify groups of similar images.
    *   Images are classified into relevant categories using `classify.py`, which builds on the embeddings.
    *   Object detection models (Grounding-DINO, Owl-ViT) are explored/utilized in `object_detection.ipynb`.
5.  **Integration:** `metadata_integrator.py` combines image analysis results with metadata, generating visualizations and reports for final insights.

## Contributing

This project is maintained for ASHA worker support initiatives. 