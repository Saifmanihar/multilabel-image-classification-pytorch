import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class MaskedBCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy loss that handles missing labels via masking"""
    def __init__(self, pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            mask: Mask indicating which labels are available (1 for available, 0 for missing)
        """
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        
        loss = loss * mask
        
        
        if mask.sum() > 0:
            return loss.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=predictions.device)

def compute_class_weights(labels_file, num_classes=4):
    """Compute class weights for handling imbalance"""
    class_counts = np.zeros(num_classes)
    total_valid_samples = 0
    
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) == 5:
            attrs = parts[1:5]  # Skip the image name
            
            for j, attr in enumerate(attrs):
                if attr != 'NA':
                    try:
                        if int(attr) == 1:
                            class_counts[j] += 1
                        total_valid_samples += 1
                    except ValueError:
                        pass
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # Compute weights 
    class_weights = total_valid_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return torch.tensor(class_weights, dtype=torch.float32)

def plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png'):
    """Plot and save loss curve as per requirements"""
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    iterations = range(1, len(train_losses) + 1)
    plt.plot(iterations, train_losses, 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation loss if available
    if val_losses:
        val_iterations = range(1, len(val_losses) + 1)
        plt.plot(val_iterations, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    # Set labels and title as per requirements
    plt.xlabel('iteration_number', fontsize=12)
    plt.ylabel('training_loss', fontsize=12)
    plt.title('Aimonk_multilabel_problem', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")

def verify_dataset(config):
    """Verify that the dataset exists and is accessible"""
    print("\n" + "="*50)
    print("VERIFYING DATASET")
    print("="*50)
    
    # Check if images folder exists
    if not os.path.exists(config.IMAGES_PATH):
        print(f" ERROR: Images folder not found: {config.IMAGES_PATH}")
        return False
    print(f" Images folder found: {config.IMAGES_PATH}")
    
    # Check if labels file exists
    if not os.path.exists(config.LABELS_FILE):
        print(f" ERROR: Labels file not found: {config.LABELS_FILE}")
        return False
    print(f" Labels file found: {config.LABELS_FILE}")
    
    # Count images in folder
    image_files = set([f for f in os.listdir(config.IMAGES_PATH) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    print(f" Found {len(image_files)} images in folder")
    
    # Check which images from labels file exist
    with open(config.LABELS_FILE, 'r') as f:
        lines = f.readlines()
    
    label_images = []
    missing = []
    
    for line in lines:
        line = line.strip()
        if line and len(line.split()) == 5:
            img_name = line.split()[0]
            label_images.append(img_name)
            if img_name not in image_files:
                missing.append(img_name)
    
    print(f" Found {len(label_images)} entries in labels file")
    if missing:
        print(f" Warning: {len(missing)} images from labels file are missing in the folder")
        print(f"   First few missing: {missing[:5]}")
        print(f"   These will be skipped during training")
    else:
        print(" All images from labels file exist in the folder")
    
    print("="*50 + "\n")
    
    return True

def calculate_metrics(predictions, targets, mask):
    """Calculate accuracy, precision, recall, and F1 score for multi-label classification"""
    predictions = (torch.sigmoid(predictions) > 0.5).float()
    
    
    predictions = predictions * mask
    targets = targets * mask
    
    # True positives, false positives, false negatives
    tp = (predictions * targets).sum().float()
    fp = (predictions * (1 - targets)).sum().float()
    fn = ((1 - predictions) * targets).sum().float()
    tn = ((1 - predictions) * (1 - targets)).sum().float()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def get_class_distribution(labels_file, num_classes=4):
    """Get the distribution of classes in the dataset"""
    class_counts = np.zeros(num_classes)
    na_counts = np.zeros(num_classes)
    
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) == 5:
            attrs = parts[1:5]
            
            for j, attr in enumerate(attrs):
                if attr == 'NA':
                    na_counts[j] += 1
                else:
                    try:
                        if int(attr) == 1:
                            class_counts[j] += 1
                    except ValueError:
                        pass
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    for i in range(num_classes):
        total = class_counts[i] + (class_counts[i] / (1 - class_counts[i]/(class_counts[i] + 1e-8)))  
        print(f"Attr{i+1}: Positive: {int(class_counts[i])}, NA: {int(na_counts[i])}")
    print("="*50)
    
    return class_counts, na_counts