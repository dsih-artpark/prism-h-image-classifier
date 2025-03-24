#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SimCLRDataset(Dataset):
    """Dataset for SimCLR that returns two augmented views of each image"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Collect all image paths
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, filename))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
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
        
        # Load the base encoder model (e.g., ResNet-50)
        if base_model == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            self.encoder_dim = 2048
        elif base_model == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder_dim = 512
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Replace the final fully connected layer
        self.encoder.fc = nn.Identity()
        
        # Add projection head
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.encoder_dim,
            output_dim=output_dim
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
        similarity_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0)) / self.temperature
        
        # Mask out the positives
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Mask out the diagnonal (self-similarity)
        negatives = similarity_matrix[self.mask].reshape(2 * self.batch_size, -1)
        
        # Create labels - positives are the "correct" predictions
        labels = torch.zeros(2 * self.batch_size).long().to(positives.device)
        
        # Calculate loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.batch_size)
        
        return loss

def train_simclr(model, train_loader, optimizer, criterion, device, epochs=100):
    """Train the SimCLR model"""
    model.train()
    losses = []
    
    for epoch in range(epochs):
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
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Record average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'simclr_checkpoint_epoch_{epoch+1}.pt')
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('SimCLR Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('simclr_loss.png')
    
    return model, losses

def get_simclr_transforms(size=224):
    """Get the augmentation transforms for SimCLR"""
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform

def save_embeddings(model, data_loader, output_file, device):
    """Extract and save embeddings from trained model"""
    model.eval()
    embeddings = []
    file_paths = []
    
    with torch.no_grad():
        for (img1, _), paths in data_loader:
            img1 = img1.to(device)
            features, _ = model(img1)
            embeddings.append(features.cpu().numpy())
            file_paths.extend(paths)
    
    embeddings = np.vstack(embeddings)
    
    # Save embeddings and file paths
    np.savez(output_file, embeddings=embeddings, file_paths=file_paths)
    print(f"Saved embeddings to {output_file}")
    
    return embeddings, file_paths

def main():
    # Configuration
    batch_size = 32
    learning_rate = 3e-4
    temperature = 0.5
    epochs = 100
    clean_data_dir = "chatgpt_results/clean"  # Directory with preprocessed images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data transforms
    train_transform = get_simclr_transforms()
    
    # Create dataset and data loader
    train_dataset = SimCLRDataset(root_dir=clean_data_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Create model
    model = SimCLRModel(base_model='resnet50', pretrained=True, output_dim=128)
    model = model.to(device)
    
    # Create loss and optimizer
    criterion = NTXentLoss(temperature=temperature, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    model, losses = train_simclr(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, 'simclr_final_model.pt')
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    main() 