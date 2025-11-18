#!/usr/bin/env python3

import pickle
from pathlib import Path  # Add this

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --- Helper Classes (SimCLRModel might be needed to load encoder easily) ---
# It's cleaner to define the necessary parts here or import from simclr.py if structured as a module


class SimCLRProjectionHead(nn.Module):  # Copied for completeness
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super(SimCLRProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLRModel(nn.Module):  # Copied for completeness
    def __init__(self, base_model="resnet50", pretrained=True, output_dim=128):
        super(SimCLRModel, self).__init__()
        if base_model == "resnet50":
            # Use the updated weights parameter
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            self.encoder_dim = 2048
        elif base_model == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.resnet18(weights=weights)
            self.encoder_dim = 512
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        self.encoder.fc = nn.Identity()
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.encoder_dim, output_dim=output_dim
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections


# --- Classification Specific Components ---


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to images.
            labels (list or np.array): List of corresponding labels (integer indices).
            transform (callable, optional): Transform to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        if len(self.image_paths) != len(self.labels):
            raise ValueError("Number of image paths must match number of labels")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            # Ensure label is a tensor if needed by the loss function
            return image, torch.tensor(label, dtype=torch.long)
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}. Skipping.")
            return None, None  # Need handling in DataLoader collate_fn or training loop
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Skipping.")
            # Handle error appropriately, maybe return None or skip
            return None, None  # Need handling in DataLoader collate_fn or training loop


class Classifier(nn.Module):
    def __init__(
        self, simclr_checkpoint_path, num_classes, freeze_encoder=True, base_model="resnet50"
    ):
        # simclr_checkpoint_path should be a Path object or string
        super(Classifier, self).__init__()

        simclr_checkpoint_path = Path(simclr_checkpoint_path)  # Ensure it's a Path

        simclr_model = SimCLRModel(base_model=base_model, pretrained=False)

        # Load the state dict from the SimCLR checkpoint
        try:
            # Use Path.exists()
            if not simclr_checkpoint_path.exists():
                raise FileNotFoundError(f"SimCLR checkpoint not found at: {simclr_checkpoint_path}")
            # torch.load works with Path objects
            checkpoint = torch.load(simclr_checkpoint_path, map_location="cpu")
            if "model_state_dict" not in checkpoint:
                raise KeyError("Checkpoint does not contain 'model_state_dict' key.")
            simclr_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded SimCLR encoder weights from: {simclr_checkpoint_path}")
        except Exception as e:
            print(f"Error loading SimCLR checkpoint {simclr_checkpoint_path}: {e}")
            raise e  # Rethrow error if loading fails

        # Extract the encoder part
        self.encoder = simclr_model.encoder
        encoder_dim = simclr_model.encoder_dim

        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("SimCLR encoder frozen.")
        else:
            print("SimCLR encoder will be fine-tuned.")

        # Add a new classification head
        self.fc = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        with torch.no_grad() if not self.encoder.fc.weight.requires_grad else torch.enable_grad():
            features = self.encoder(x)
        output = self.fc(features)
        return output


# --- Training Loop ---


def train_classifier(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    checkpoint_dir="classification_checkpoints",
    patience=5,
):
    checkpoint_dir = Path(checkpoint_dir)  # Convert to Path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir()
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_path = checkpoint_dir / "best_classifier.pt"  # Use / operator

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=patience // 2, verbose=True
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (images, labels) in enumerate(progress_bar):
            # Skip batch if collate_fn returned None
            if images is None or labels is None:
                print(f"Warning: Skipping empty batch {i} in training.")
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            progress_bar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

        # Ensure train_targets is not empty before calculating metrics
        if not train_targets:
            print(
                f"Warning: No valid training samples found in Epoch {epoch+1}. Skipping epoch metrics."
            )
            continue

        epoch_loss = running_loss / len(
            train_targets
        )  # Use len(train_targets) as it reflects actual processed samples
        epoch_acc = accuracy_score(train_targets, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if images is None or labels is None:
                    print("Warning: Skipping empty batch in validation.")
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Ensure val_targets is not empty
        if not val_targets:
            print(
                f"Warning: No valid validation samples processed in Epoch {epoch+1}. Cannot calculate validation metrics."
            )
            val_loss_epoch = float("inf")
            val_acc = 0.0
        else:
            val_loss_epoch = val_loss / len(val_targets)
            val_acc = accuracy_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)

        # Save best model based on validation accuracy & Early stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                best_model_path,
            )
            print(f"Saved new best model to {best_model_path} (Val Acc: {best_val_accuracy:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training finished. Best Validation Accuracy: {best_val_accuracy:.4f}")
    # Load best model for final return/evaluation
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from {best_model_path}")
    except FileNotFoundError:
        print(
            f"Warning: Best model checkpoint {best_model_path} not found. Returning the last state model."
        )
    except Exception as e:
        print(
            f"Warning: Error loading best model checkpoint {best_model_path}: {e}. Returning the last state model."
        )

    return model


# --- Collate Function --- Doesn't appear to be used ---


def collate_fn(batch):
    # Filter out items where the image is None (due to loading errors)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        # Return None values that will be checked and skipped in the training loop
        return None, None
    # Use default collate for the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)


# --- Main Execution Logic ---


def classification_main():
    # --- Configuration ---
    simclr_output_dir = Path("simclr_finetuned")  # Use Path
    # !! IMPORTANT: Verify this path points to the correct checkpoint from SimCLR training !!
    simclr_checkpoint_path = simclr_output_dir / "checkpoints" / "best_model.pt"  # Use Path and /
    splits_file = simclr_output_dir / "data_splits.pkl"  # Use Path and /

    # !! IMPORTANT: Path to your REFINED labels !!
    # This needs to be created based on your clustering analysis.
    refined_labels_file = Path("refined_labels.csv")  # Use Path <--- CREATE THIS FILE

    num_classes = 10  # <--- SET THIS: Number of container types (or other classes)
    batch_size = 32
    learning_rate = 1e-4  # Start lower for fine-tuning
    epochs = 50  # Adjust as needed
    freeze_encoder = True  # Freeze encoder initially, maybe unfreeze later for fine-tuning
    classification_checkpoint_dir = Path("classification_checkpoints")  # Use Path
    num_workers = 0  # Set to 0 for macOS compatibility if needed
    early_stopping_patience = 10  # Patience for early stopping
    # --- End Configuration ---

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data splits
    try:
        # Use Path.open()
        with splits_file.open("rb") as f:
            splits = pickle.load(f)
        train_paths = splits["train"]
        val_paths = splits["val"]
        test_paths = splits["test"]
        print(f"Loaded data splits from {splits_file}")
    except FileNotFoundError:
        print(f"Error: Data splits file not found at {splits_file}")
        print("Please ensure 'simclr.py' or 'extract_embeddings.py' generated the splits file.")
        return
    except Exception as e:
        print(f"Error loading splits file {splits_file}: {e}")
        return

    # --- Load Refined Labels (EXAMPLE) ---
    # You MUST adapt this part based on how you save refined labels after clustering
    path_to_label = {}
    try:
        # Use Path.exists()
        if not refined_labels_file.exists():
            raise FileNotFoundError(f"Refined labels file not found at {refined_labels_file}")

        # pd.read_csv generally accepts Path objects directly
        label_df = pd.read_csv(refined_labels_file)
        # Basic validation of the CSV file
        if "file_path" not in label_df.columns or "refined_label" not in label_df.columns:
            raise ValueError(
                f"CSV file {refined_labels_file} must contain 'file_path' and 'refined_label' columns."
            )

        # Ensure labels are integers
        if not pd.api.types.is_integer_dtype(label_df["refined_label"]):
            print("Warning: 'refined_label' column is not integer type. Attempting conversion.")
            label_df["refined_label"] = label_df["refined_label"].astype(int)

        # Normalize file paths in the CSV for better matching (optional, depends on how paths were saved)
        # label_df['file_path'] = label_df['file_path'].apply(os.path.normpath)
        path_to_label = pd.Series(label_df.refined_label.values, index=label_df.file_path).to_dict()
        print(f"Loaded refined labels for {len(path_to_label)} images from {refined_labels_file}")

        # Check if the number of classes derived from labels matches the configured num_classes
        actual_num_classes = label_df["refined_label"].nunique()
        if actual_num_classes != num_classes:
            print(
                f"Warning: Configured num_classes ({num_classes}) does not match the number of unique labels found in {refined_labels_file} ({actual_num_classes}). Using {actual_num_classes} classes found in the label file."
            )
            num_classes = actual_num_classes  # Adjust num_classes based on data

    except FileNotFoundError:
        print(f"Error: Refined labels file not found at '{refined_labels_file}'.")
        print(
            "Please create this file (e.g., a CSV with 'file_path' and 'refined_label' columns) based on your clustering analysis."
        )
        return
    except ValueError as e:
        print(f"Error processing refined labels file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading refined labels: {e}")
        return

    # Map paths to labels for each split and filter missing labels
    def get_filtered_split(paths):
        filtered_paths = []
        filtered_labels = []
        missing_count = 0
        for p_str in paths:  # Keep paths as strings as loaded from pickle for now
            # Use the string path directly for dictionary lookup
            label = path_to_label.get(p_str)
            # If paths in CSV might be relative/absolute differently, normalization might be needed
            # (Normalization usually done when CREATING the label file and loading splits)
            # Example if needed: p_norm = Path(p_str).resolve() or os.path.normpath(p_str)
            # label = path_to_label.get(str(p_norm))
            if label is not None:
                filtered_paths.append(p_str)  # Keep as string for Dataset
                filtered_labels.append(label)
            else:
                missing_count += 1
        if missing_count > 0:
            print(
                f"Warning: Could not find labels for {missing_count} images in this split. They will be excluded."
            )
        return filtered_paths, filtered_labels

    print("Mapping labels to data splits...")
    train_paths_filt, train_labels_filt = get_filtered_split(train_paths)
    val_paths_filt, val_labels_filt = get_filtered_split(val_paths)
    test_paths_filt, test_labels_filt = get_filtered_split(test_paths)

    if not train_paths_filt or not val_paths_filt:
        print(
            "Error: No training or validation samples remaining after label mapping. Cannot proceed."
        )
        return

    print(
        f"Using {len(train_paths_filt)} train, {len(val_paths_filt)} val, {len(test_paths_filt)} test samples with labels."
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Slightly stronger augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Add color jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    try:
        train_dataset = ClassificationDataset(
            train_paths_filt, train_labels_filt, transform=train_transform
        )
        val_dataset = ClassificationDataset(
            val_paths_filt, val_labels_filt, transform=eval_transform
        )
        test_dataset = (
            ClassificationDataset(test_paths_filt, test_labels_filt, transform=eval_transform)
            if test_paths_filt
            else None
        )
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device != torch.device("cpu") else False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device != torch.device("cpu") else False,
        collate_fn=collate_fn,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device != torch.device("cpu") else False,
            collate_fn=collate_fn,
        )
        if test_dataset
        else None
    )

    # Initialize model
    try:
        model = Classifier(
            simclr_checkpoint_path=simclr_checkpoint_path,
            num_classes=num_classes,  # Use the potentially adjusted num_classes
            freeze_encoder=freeze_encoder,
        ).to(device)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
        print(f"Error initializing the classification model: {e}")
        print("Please check the SimCLR checkpoint path and its contents.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model initialization: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    # Only optimize the classifier head if encoder is frozen
    parameters_to_optimize = model.fc.parameters() if freeze_encoder else model.parameters()
    optimizer = optim.AdamW(
        parameters_to_optimize, lr=learning_rate, weight_decay=1e-4
    )  # Use AdamW

    # Train the model
    print(f"Starting classifier training for {num_classes} classes...")
    trained_model = train_classifier(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs,
        classification_checkpoint_dir,
        patience=early_stopping_patience,
    )

    # Evaluate on test set if available
    if test_loader:
        print("Evaluating on test set...")
        trained_model.eval()
        test_preds, test_targets = [], []
        test_filepaths = []  # Store filepaths for potential error analysis

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader, desc="[Test]")):
                if images is None or labels is None:
                    print(f"Warning: Skipping empty batch {i} in testing.")
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = trained_model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
                # Assuming test_dataset stores paths at index 0
                # test_filepaths.extend([test_dataset.image_paths[idx] for idx in test_loader.sampler.indices[i*batch_size:(i+1)*batch_size]]) # Complicated way if using sampler

        if not test_targets:
            print("No valid test samples were processed. Cannot evaluate test accuracy.")
        else:
            test_acc = accuracy_score(test_targets, test_preds)
            print(f"Final Test Accuracy: {test_acc:.4f}")

            # Print classification report for more details
            print("\nTest Set Classification Report:")
            try:
                # Get class labels if possible (e.g., from a mapping file or sorted unique labels)
                target_names = [f"Class {i}" for i in range(num_classes)]
                print(
                    classification_report(
                        test_targets, test_preds, target_names=target_names, zero_division=0
                    )
                )
            except Exception as e:
                print(f"Could not generate detailed classification report: {e}")
                # Fallback to basic report without names
                print(classification_report(test_targets, test_preds, zero_division=0))
    else:
        print("No test set data available for final evaluation.")


if __name__ == "__main__":
    classification_main()
