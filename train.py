import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

from config import parse_arguments, create_config_from_args
from dataset import create_dataloaders
from models import MultiLabelModel
from utils import MaskedBCEWithLogitsLoss, compute_class_weights, plot_loss_curve, verify_dataset

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(loader, desc='Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['mask'].to(device)
        
       
        optimizer.zero_grad()
        
        
        outputs = model(images)
        
        
        loss = criterion(outputs, labels, mask)
        
        
        loss.backward()
        
        
        optimizer.step()
        
        
        running_loss += loss.item()
        
       
        pbar.set_postfix({'loss': running_loss / len(loader)})
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            
           
            outputs = model(images)
            
            
            loss = criterion(outputs, labels, mask)
            
            running_loss += loss.item()
    
    return running_loss / len(loader)

def main():
    
    args = parse_arguments()
    config = create_config_from_args(args)
    
    
    print("\n" + "="*60)
    print("MULTI-LABEL CLASSIFICATION TRAINING")
    print("="*60)
    print(f"Data path: {config.BASE_PATH}")
    print(f"Images path: {config.IMAGES_PATH}")
    print(f"Labels file: {config.LABELS_FILE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*60)
    
    # Verify dataset exists
    if not verify_dataset(config):
        print("Error: Dataset verification failed. Please check the paths.")
        return
    
    # Compute class weights for handling imbalance
    print("Computing class weights...")
    class_weights = compute_class_weights(config.LABELS_FILE, config.NUM_CLASSES)
    class_weights = class_weights.to(config.DEVICE)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"\nCreating model: {config.MODEL_NAME}")
    model = MultiLabelModel(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True  # Always use pretrained weights for fine-tuning
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    criterion = MaskedBCEWithLogitsLoss(pos_weight=class_weights)
    
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    
    scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
    
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'class_weights': class_weights,
                'config': config
            }, config.MODEL_SAVE_PATH)
            print(f"✓ Saved best model with val_loss: {val_loss:.4f}")
    
    # Plot loss curve
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved at: {config.MODEL_SAVE_PATH}")
    
    print("\nPlotting loss curve...")
    plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png')
    print("Loss curve saved as: loss_curve.png")

if __name__ == "__main__":
    main()