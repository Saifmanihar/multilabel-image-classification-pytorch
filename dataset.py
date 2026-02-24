import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class MultiLabelDataset(Dataset):
    def __init__(self, images_path, labels_file, transform=None):
        self.images_path = images_path
        self.transform = transform
        
        
        self.data = self._parse_labels_file(labels_file)
        
    def _parse_labels_file(self, labels_file):
        """Parse the custom labels.txt file format - each line: image_name attr1 attr2 attr3 attr4"""
        data = []
        missing_images = []
        
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            
            parts = line.split()
            
           
            if len(parts) == 5:
                image_name = parts[0]
                
                
                image_path = os.path.join(self.images_path, image_name)
                if not os.path.exists(image_path):
                    missing_images.append(image_name)
                    continue  # Skip this image if it doesn't exist
                
                attr1 = parts[1]
                attr2 = parts[2]
                attr3 = parts[3]
                attr4 = parts[4]
                
                
                labels = []
                mask = []  
                
                for attr in [attr1, attr2, attr3, attr4]:
                    if attr == 'NA':
                        labels.append(0)  
                        mask.append(0)    
                    else:
                        try:
                            labels.append(int(attr))
                            mask.append(1)  
                        except ValueError:
                            # Handle invalid values
                            print(f"Warning: Invalid value '{attr}' for {image_name}, treating as NA")
                            labels.append(0)
                            mask.append(0)
                
                data.append({
                    'image_name': image_name,
                    'labels': torch.tensor(labels, dtype=torch.float32),
                    'mask': torch.tensor(mask, dtype=torch.float32)
                })
            else:
                print(f"Warning: Skipping malformed line: {line}")
        
        print(f"Loaded {len(data)} samples from labels file")
        if missing_images:
            print(f"Warning: Skipped {len(missing_images)} images that don't exist in the folder")
            print(f"First few missing: {missing_images[:5]}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        
        image_path = os.path.join(self.images_path, item['image_name'])
        image = Image.open(image_path).convert('RGB')
        
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'labels': item['labels'],
            'mask': item['mask'],
            'image_name': item['image_name']
        }

def get_transforms(img_size=224, mean=None, std=None, is_train=True):
    """Get transforms with augmentation for training"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if is_train:
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform

def create_dataloaders(config):
    """Create train and validation dataloaders"""
    # Create datasets with appropriate transforms
    train_transform = get_transforms(config.IMG_SIZE, config.MEAN, config.STD, is_train=True)
    val_transform = get_transforms(config.IMG_SIZE, config.MEAN, config.STD, is_train=False)
    
    # Load full dataset
    full_dataset = MultiLabelDataset(
        config.IMAGES_PATH, 
        config.LABELS_FILE, 
        transform=train_transform  # Will be overridden for validation
    )
    
    # Split dataset (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Override transforms for each split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader