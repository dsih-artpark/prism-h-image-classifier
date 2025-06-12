#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

# --- Helper Classes (Copied from simclr.py for standalone execution) ---

class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR"""
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(SimCLRProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.projection(x)

class SimCLRModel(nn.Module):
    """SimCLR model with encoder and projection head"""
    def __init__(self, base_model='resnet50', pretrained=True, output_dim=128):
        super(SimCLRModel, self).__init__()
        if base_model == 'resnet50':
            # Use the updated weights parameter for torchvision >= 0.13
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            self.encoder_dim = 2048
        elif base_model == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.resnet18(weights=weights)
            self.encoder_dim = 512
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        self.encoder.fc = nn.Identity()
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.encoder_dim,
            output_dim=output_dim
        )
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        # Note: For embedding extraction, we only need 'features'
        return features, projections # Original return for compatibility if needed elsewhere

class PathBasedDataset(Dataset): # Simplified dataset for extraction
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path # Return image and its path
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}")
            # Return None or a placeholder if an image fails to load
            # For simplicity, we'll return None and handle it in the loop
            return None, img_path

# --- Main Extraction Logic ---

def extract_embeddings_main():
    # --- Configuration ---
    output_dir = Path("simclr_finetuned")
    checkpoint_dir = output_dir / "checkpoints"
    splits_file = output_dir / "data_splits.pkl"
    batch_size = 64 # Can be larger for inference
    num_workers = 0 # Set to 0 for macOS compatibility if needed
    # --- End Configuration ---

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load data splits
    try:
        with splits_file.open('rb') as f:
            splits = pickle.load(f)
        train_paths = splits['train']
        val_paths = splits['val']
        test_paths = splits['test']
        all_image_paths = train_paths + val_paths + test_paths
        print(f"Loaded data splits from {splits_file}")
    except FileNotFoundError:
        print(f"Error: Data splits file not found at {splits_file}")
        print("Please ensure 'simclr.py' has been run successfully to generate the splits file.")
        return
    except Exception as e:
        print(f"Error loading splits file {splits_file}: {e}")
        return

    # Initialize model architecture
    # Set pretrained=False as we are loading specific weights
    model = SimCLRModel(base_model='resnet50', pretrained=False, output_dim=128)

    # Find the best checkpoint to load
    best_model_path = checkpoint_dir / 'best_model.pt'
    latest_epoch_checkpoint = None

    # Check if checkpoint directory exists
    if not checkpoint_dir.is_dir():
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        print("Please ensure 'simclr.py' has been run and checkpoints are saved.")
        return

    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

    if checkpoints:
        # Sort by epoch number extracted from filename (Path.stem extracts filename without suffix)
        try:
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
            latest_epoch_checkpoint = checkpoints[0]
        except (ValueError, IndexError):
            print("Warning: Could not parse epoch number from checkpoint filenames.")
            latest_epoch_checkpoint = None # Reset if parsing fails

    checkpoint_to_load = None
    if best_model_path.exists():
        checkpoint_to_load = best_model_path
        print(f"Found best model checkpoint: {best_model_path}")
    elif latest_epoch_checkpoint:
        checkpoint_to_load = latest_epoch_checkpoint
        print(f"Using latest epoch checkpoint: {latest_epoch_checkpoint}")
    else:
        print(f"Error: No suitable checkpoint (.pt file starting with 'checkpoint_epoch_' or 'best_model.pt') found in {checkpoint_dir}")
        print("Please ensure 'simclr.py' has run and saved checkpoints.")
        return

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        # Ensure the checkpoint contains the model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded weights from {checkpoint_to_load}")
        else:
            print(f"Error: Checkpoint {checkpoint_to_load} does not contain 'model_state_dict'.")
            return
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_to_load}")
        return
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_to_load}: {e}")
        return

    model = model.to(device)
    model.eval() # Set to evaluation mode

    # Define evaluation transform (consistent preprocessing)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to extract embeddings for a given set of paths
    def run_extraction(paths, output_filename):
        dataset = PathBasedDataset(paths, transform=eval_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device != torch.device('cpu') else False, # pin_memory only works with CUDA
            drop_last=False
        )

        embeddings_list = []
        filepaths_list = []

        with torch.no_grad():
            for images, loaded_paths in tqdm(dataloader, desc=f"Extracting for {output_filename}"):
                # Handle potential loading errors from dataset
                # Filter out None images and their corresponding paths
                valid_indices = [i for i, img in enumerate(images) if img is not None]
                if not valid_indices: # Skip batch if all images failed to load
                    # loaded_paths is a tuple here, need to access elements
                    failed_paths = [p for i, p in enumerate(loaded_paths) if i not in valid_indices]
                    if failed_paths: # Only print if there were actually paths that failed
                        print(f"Warning: Skipping batch, failed to load images: {failed_paths}")
                    continue

                images_tensor = torch.stack([images[i] for i in valid_indices]).to(device)
                valid_paths = [loaded_paths[i] for i in valid_indices]

                # Get features from the encoder
                features, _ = model(images_tensor)

                embeddings_list.append(features.cpu().numpy())
                filepaths_list.extend(valid_paths)

        if embeddings_list:
            embeddings_np = np.vstack(embeddings_list)
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename
            # Save file paths as UTF-8 strings
            np.savez_compressed(output_path, embeddings=embeddings_np, file_paths=np.array(filepaths_list, dtype='str'))
            print(f"Saved {len(embeddings_np)} embeddings to {output_path}")
        else:
            print(f"No embeddings extracted for {output_filename}. This might happen if all images in the split failed to load.")

    # Run extraction for all required splits
    print("Starting embedding extraction...")
    run_extraction(train_paths, 'train_embeddings.npz')
    run_extraction(val_paths, 'val_embeddings.npz')
    run_extraction(test_paths, 'test_embeddings.npz')
    run_extraction(all_image_paths, 'all_embeddings.npz')

    print("Embedding extraction complete.")

if __name__ == "__main__":
    extract_embeddings_main() 