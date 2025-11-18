#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SimCLRDataset(Dataset):
    """Dataset for SimCLR that returns two augmented views of each image"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string or Path): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []

        # Collect all image paths
        for entry in self.root_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                self.image_paths.append(entry)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # Apply the same transform twice to get two different augmented views
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2

        return image, image


class SimCLRProjectionHead(nn.Module):
    """Projection head for SimCLR"""

    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(SimCLRProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLRModel(nn.Module):
    """SimCLR model with encoder and projection head"""

    def __init__(self, base_model="resnet50", pretrained=True, output_dim=128):
        super(SimCLRModel, self).__init__()

        if base_model == "resnet50":
            self.encoder = models.resnet50(pretrained=pretrained)
            self.encoder_dim = 2048
        elif base_model == "resnet18":
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder_dim = 512
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        # Replace the final fully connected layer
        self.encoder.fc = nn.Identity()

        # Add projection head
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.encoder_dim, output_dim=output_dim
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, F.normalize(projections, dim=1)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    """

    def __init__(self, temperature=0.5, batch_size=32):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # Mask to remove positive examples from the denominator of the loss function
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        self.register_buffer("mask", mask)

    def forward(self, z_i, z_j):
        """
        Calculate NT-Xent loss
        Args:
            z_i, z_j: Normalized projection vectors from the two augmented views
        """
        # Calculate cosine similarity
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = (
            self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))
            / self.temperature
        )

        # Mask out the positives
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Mask out the diagnonal (self-similarity)
        negatives = similarity_matrix[self.mask].reshape(2 * self.batch_size, -1)

        labels = torch.zeros(2 * self.batch_size).long().to(positives.device)

        # Calculate loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.batch_size)

        return loss


def train_simclr(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs=100,
    checkpoint_dir="checkpoints",
    resume_from=None,
    early_stopping_patience=10,
    validation_loader=None,
    writer=None,
):
    """Train the SimCLR model with proper analytics and early stopping"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trackers
    best_loss = float("inf")
    best_model_path = checkpoint_dir / "best_model.pt"
    patience_counter = 0
    start_epoch = 0
    train_losses = []
    val_losses = []
    lr_history = []
    global_step = 0

    # Resume from checkpoint if available
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            if "best_loss" in checkpoint:
                best_loss = checkpoint["best_loss"]
            if "train_losses" in checkpoint:
                train_losses = checkpoint["train_losses"]
            if "val_losses" in checkpoint:
                val_losses = checkpoint["val_losses"]
            if "lr_history" in checkpoint:
                lr_history = checkpoint["lr_history"]
            if "global_step" in checkpoint:
                global_step = checkpoint["global_step"]
            print(
                f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}, global step: {global_step}"
            )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Log model graph to TensorBoard if available
    if writer is not None:
        # Get a batch of data for graph visualization
        sample_images, _ = next(iter(train_loader))
        sample_images = sample_images.to(device)
        writer.add_graph(model, sample_images)

        # Log some example augmented pairs
        fig = plt.figure(figsize=(12, 6))
        for i in range(min(4, len(sample_images))):
            ax1 = fig.add_subplot(2, 4, i + 1)
            img = sample_images[i].cpu().permute(1, 2, 0).numpy()
            # Denormalize image
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax1.imshow(img)
            ax1.set_title(f"View 1 - img {i}")
            ax1.axis("off")

            sample2, _ = next(iter(train_loader))
            ax2 = fig.add_subplot(2, 4, i + 5)
            img2 = sample2[i].cpu().permute(1, 2, 0).numpy()
            img2 = img2 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img2 = np.clip(img2, 0, 1)
            ax2.imshow(img2)
            ax2.set_title(f"View 2 - img {i}")
            ax2.axis("off")

        plt.tight_layout()
        writer.add_figure("Example Augmented Pairs", fig, global_step=0)

    # Function to evaluate on validation set
    def evaluate():
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images1, images2 in validation_loader:
                images1, images2 = images1.to(device), images2.to(device)
                _, z1 = model(images1)
                _, z2 = model(images2)
                loss = criterion(z1, z2)
                total_val_loss += loss.item()
        return total_val_loss / len(validation_loader)

    # Function to log embeddings
    def log_embeddings(step):
        if writer is None:
            return

        # Extract embeddings for visualization
        model.eval()
        embeddings = []
        imgs = []
        with torch.no_grad():
            for i, (images1, _) in enumerate(validation_loader):
                if i >= 2:  # Limit to a few batches for visualization
                    break
                images1 = images1.to(device)
                features, _ = model(images1)
                embeddings.append(features.cpu().numpy())
                imgs.append(images1.cpu())

        if not embeddings:
            return

        embeddings = np.vstack(embeddings)
        imgs = torch.cat(imgs, dim=0)

        # Use t-SNE for dimensionality reduction
        if len(embeddings) > 10:  # Need enough samples for meaningful t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)

            # Plot t-SNE visualization
            fig = plt.figure(figsize=(10, 10))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
            plt.title("t-SNE of Feature Embeddings")
            writer.add_figure("Embeddings/t-SNE", fig, global_step=step)

        # Log embeddings with images
        writer.add_embedding(
            mat=torch.from_numpy(embeddings),
            label_img=imgs,
            global_step=step,
            tag="features",  # Explicit tag for embeddings
        )

    print(f"Starting training from epoch {start_epoch+1}/{epochs}")
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for i, (images1, images2) in enumerate(progress_bar):
            # Move images to device
            images1 = images1.to(device)
            images2 = images2.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass for both augmented views
            _, z1 = model(images1)
            _, z2 = model(images2)

            # Calculate loss
            loss = criterion(z1, z2)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # Log to TensorBoard (every 10 batches)
            if writer is not None and i % 10 == 0:
                writer.add_scalar("Batch/train_loss", loss.item(), global_step)
                global_step += 1

        # Record average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        # Validation phase (if validation loader provided)
        val_loss = None
        if validation_loader:
            val_loss = evaluate()
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
            )

            # Log metrics to TensorBoard
            if writer is not None:
                writer.add_scalar("Epoch/train_loss", epoch_loss, epoch)
                writer.add_scalar("Epoch/val_loss", val_loss, epoch)
                writer.add_scalar("Epoch/learning_rate", current_lr, epoch)

                # Log embeddings periodically
                if epoch % 5 == 0 or epoch == epochs - 1:
                    log_embeddings(epoch)

            # Update scheduler based on validation loss
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "best_loss": best_loss,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "lr_history": lr_history,
                        "global_step": global_step,
                    },
                    best_model_path,
                )
                patience_counter = 0
                print(f"New best model saved with validation loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}"
                )

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    break
        else:
            # If no validation set, use training loss
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar("Epoch/train_loss", epoch_loss, epoch)
                writer.add_scalar("Epoch/learning_rate", current_lr, epoch)

                # Log embeddings periodically
                if epoch % 5 == 0 or epoch == epochs - 1:
                    log_embeddings(epoch)

            # Save if better than best so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss,
                        "best_loss": best_loss,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "lr_history": lr_history,
                        "global_step": global_step,
                    },
                    best_model_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    break

            # Update scheduler based on training loss
            scheduler.step(epoch_loss)

        # Regular checkpoint (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss if validation_loader else epoch_loss,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "lr_history": lr_history,
                    "global_step": global_step,
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

    # Plot training curves
    plt.figure(figsize=(15, 5))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SimCLR Training Loss")
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot(lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "training_curves.png")

    # Save training history as CSV for further analysis
    history = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses if val_losses else [None] * len(train_losses),
            "learning_rate": lr_history,
        }
    )
    history.to_csv(checkpoint_dir / "training_history.csv", index=False)

    print(f"Training completed. Best loss: {best_loss:.4f}")
    print(f"Training analytics saved to {checkpoint_dir}/")

    return model, {"train_losses": train_losses, "val_losses": val_losses, "lr_history": lr_history}


def get_simclr_transforms(size=224):
    """Get the augmentation transforms for SimCLR"""
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    # Calculate kernel size and make sure it's odd
    kernel_size = int(0.1 * size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3  # Minimum kernel size

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform


def save_embeddings(model, data_loader, output_file, device):
    """Extract and save embeddings from trained model"""
    output_file = Path(output_file)
    model.eval()
    embeddings = []
    file_paths = []  # Keep paths as Path objects internally

    with torch.no_grad():
        # Assuming data_loader yields ((img1, img2), path) or similar
        for item in tqdm(data_loader, desc="Extracting embeddings"):
            # Adapt based on actual data_loader structure
            if isinstance(item, (tuple, list)) and len(item) == 2:
                # Handle cases like ((img1, img2), path) or (img_tuple, path)
                if isinstance(item[0], (tuple, list)) and len(item[0]) >= 1:
                    img_data = item[0][0]  # Get the first image view for features
                else:
                    img_data = item[0]  # Assume item[0] is the image tensor

                path_data = item[1]
            else:
                # Fallback or error handling if structure is unexpected
                print(f"Unexpected item structure in data_loader: {type(item)}")
                continue  # Or raise an error

            img_data = img_data.to(device)
            features, _ = model(img_data)
            embeddings.append(features.cpu().numpy())

            # Ensure paths are handled correctly (list or single Path)
            if isinstance(path_data, list):
                file_paths.extend([Path(p) for p in path_data])
            else:
                file_paths.append(Path(path_data))

    if not embeddings:
        print(f"No embeddings extracted for {output_file}")
        return None, None

    embeddings = np.vstack(embeddings)

    # Save embeddings and convert file paths to strings for npz compatibility
    np.savez(str(output_file), embeddings=embeddings, file_paths=[str(p) for p in file_paths])
    print(f"Saved {len(embeddings)} embeddings to {output_file}")

    return embeddings, file_paths  # Return original Path objects if needed downstream


class EmbeddingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []

        for entry in self.root_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                self.image_paths.append(entry)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (image, image), img_path  # Return Path object


class PathBasedSimCLRDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [Path(p) for p in image_paths]  # Ensure paths are Path objects
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
            # Return path along with images for save_embeddings
            return (view1, view2), img_path

        # Return path even if no transform
        return (image, image), img_path


def main():
    # Configuration
    batch_size = 32
    learning_rate = 3e-4
    temperature = 0.5
    epochs = 100  # Maximum epochs (early stopping may trigger sooner)
    clean_data_dir = Path("preprocess_results/clean")  # Updated path to preprocessed clean images
    output_dir = Path("simclr_finetuned")
    checkpoint_dir = output_dir / "checkpoints"
    early_stopping_patience = 10  # Stop after this many epochs without improvement
    tensorboard_dir = output_dir / "tensorboard"

    # Data split ratios - proper methodology to avoid leakage
    train_ratio = 0.7  # 70% for training
    val_ratio = 0.15  # 15% for validation
    test_ratio = 0.15  # 15% for testing

    # Ratio for validation during fine-tuning (taken from training set)
    finetune_val_ratio = 0.1  # 10% of training data for validation during fine-tuning

    # Determine number of workers (reduced for macOS)
    num_workers = 0  # Set to 0 for macOS to avoid multiprocessing issues

    # Device configuration for Mac (MPS) or CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU (GPU not available)")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_subdir = tensorboard_dir / current_time
    writer = SummaryWriter(log_dir=tensorboard_subdir)  # SummaryWriter accepts Path
    print(f"TensorBoard logs will be saved to {tensorboard_subdir}")
    print(f"To view TensorBoard, run: tensorboard --logdir={tensorboard_dir}")

    # Record hyperparameters in TensorBoard
    hyperparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "finetune_val_ratio": finetune_val_ratio,
        "early_stopping_patience": early_stopping_patience,
        "device": str(device),
    }
    # Log hyperparameters
    writer.add_hparams(hyperparams, {"status": 1})

    train_transform = get_simclr_transforms()

    # Collect all image paths using pathlib
    all_image_paths = []
    for entry in clean_data_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            all_image_paths.append(entry)

    print(f"Found {len(all_image_paths)} images in {clean_data_dir}")

    from sklearn.model_selection import train_test_split

    # First split out the test set
    train_val_paths, test_paths = train_test_split(
        all_image_paths, test_size=test_ratio, random_state=42
    )

    # Then split the remaining data into train and validation
    train_paths, val_paths = train_test_split(
        train_val_paths, test_size=val_ratio / (train_ratio + val_ratio), random_state=42
    )

    # Save the splits for downstream tasks
    splits = {"train": train_paths, "val": val_paths, "test": test_paths}

    # Save splits to file using pathlib
    import pickle

    with (output_dir / "data_splits.pkl").open("wb") as f:
        pickle.dump(splits, f)

    # For fine-tuning, further split the training set to monitor performance
    finetune_train_paths, finetune_val_paths = train_test_split(
        train_paths, test_size=finetune_val_ratio, random_state=42
    )

    print(
        f"Data splits: {len(finetune_train_paths)} for SimCLR training, "
        f"{len(finetune_val_paths)} for SimCLR validation, "
        f"{len(val_paths)} held-out validation, "
        f"{len(test_paths)} held-out test"
    )

    train_dataset = PathBasedSimCLRDataset(finetune_train_paths, transform=train_transform)
    val_dataset = PathBasedSimCLRDataset(finetune_val_paths, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for SimCLR loss calculation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for NTXent loss
    )

    print(
        f"SimCLR fine-tuning dataset: {len(train_dataset)} training images, {len(val_dataset)} validation images"
    )

    model = SimCLRModel(base_model="resnet50", pretrained=True, output_dim=128)

    # Check for existing checkpoints in order of priority using pathlib
    latest_checkpoint = None
    best_model_path = checkpoint_dir / "best_model.pt"

    # First check if best model exists
    if best_model_path.exists():
        latest_checkpoint = best_model_path
        print(f"Found best model checkpoint: {best_model_path}")
    else:
        # Otherwise find the latest checkpoint using glob and sorting
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            # Sort by epoch number extracted from filename
            checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]), reverse=True)
            latest_checkpoint = checkpoints[0]
            print(f"Found latest checkpoint: {latest_checkpoint}")

    # If no checkpoints found, look for pretrained SimCLR weights
    if not latest_checkpoint:
        pretrained_path = Path("pretrained_simclr.pt")  # Use Path
        if pretrained_path.exists():
            print(f"Using pretrained SimCLR weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded pretrained weights")
        else:
            print("No checkpoints or pretrained weights found, using ImageNet weights")

    # Move model to device
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = NTXentLoss(temperature=temperature, batch_size=batch_size)

    # Train the model
    print(f"Starting SimCLR fine-tuning on {len(train_dataset)} train images")
    model, history = train_simclr(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        resume_from=latest_checkpoint,
        early_stopping_patience=early_stopping_patience,
        validation_loader=val_loader,
        writer=writer,  # Pass the TensorBoard writer
    )

    # Save the final fine-tuned model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        output_dir / "simclr_finetuned_final.pt",
    )  # Use Path object

    # Extract and save embeddings for all splits using the best model
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Function to extract embeddings for a given set of paths
    def extract_embeddings(paths: list[Path], output_file: Path):
        dataset = PathBasedSimCLRDataset(paths, transform=eval_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,  # Include all samples
        )

        # Use the modified save_embeddings function
        # It now handles Path objects and saves paths as strings in the npz file
        embeddings, _ = save_embeddings(model, dataloader, output_file, device)
        return embeddings  # Return only embeddings as paths are saved internally

    # Extract embeddings for each split using pathlib paths
    print("Extracting embeddings for all data splits...")
    train_embeddings = extract_embeddings(train_paths, output_dir / "train_embeddings.npz")
    val_embeddings = extract_embeddings(val_paths, output_dir / "val_embeddings.npz")
    test_embeddings = extract_embeddings(test_paths, output_dir / "test_embeddings.npz")

    # Also save all embeddings together for convenience
    all_embeddings = extract_embeddings(all_image_paths, output_dir / "all_embeddings.npz")

    # Visualize embeddings in TensorBoard
    if train_embeddings is not None and writer is not None:
        try:
            # Use t-SNE for visualization
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(
                train_embeddings[:1000]
            )  # Limit to 1000 for visualization

            # Create scatter plot
            fig = plt.figure(figsize=(10, 10))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
            plt.title("t-SNE of Final Embeddings")
            writer.add_figure("Final Embeddings/t-SNE", fig)
            print("Added final embedding visualization to TensorBoard")
        except Exception as e:
            print(f"Failed to visualize embeddings: {e}")

    print(f"Fine-tuning complete. Model and embeddings saved to {output_dir}/")
    print(
        f"Training analytics available at {checkpoint_dir}/training_curves.png and {checkpoint_dir}/training_history.csv"
    )
    print(f"TensorBoard logs available at {tensorboard_subdir}")
    print(f"To view: tensorboard --logdir={tensorboard_dir}")
    print(f"Data splits saved to {output_dir}/data_splits.pkl for use in downstream tasks")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
