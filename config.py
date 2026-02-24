import torch
import os
import argparse

class Config:
    def __init__(self, data_path=None, images_folder="images", labels_file="label.txt"):
        if data_path:
            self.BASE_PATH = data_path
        else:
            self.BASE_PATH = "."  # Default to current directory
        
        self.IMAGES_PATH = os.path.join(self.BASE_PATH, images_folder)
        self.LABELS_FILE = os.path.join(self.BASE_PATH, labels_file)
        self.MODEL_SAVE_PATH = os.path.join(self.BASE_PATH, "best_model.pth")
        
        # Model parameters
        self.MODEL_NAME = "resnet50" 
        self.NUM_CLASSES = 4
        self.PRETRAINED = True
        
        # Training parameters
        self.BATCH_SIZE = 32
        self.EPOCHS = 30
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.01
        self.NUM_WORKERS = 4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
       
        self.IMG_SIZE = 224  # Standard ImageNet size
        self.MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
        self.STD = [0.229, 0.224, 0.225]   # ImageNet std
        
        
        self.CLASS_NAMES = ["Attr1", "Attr2", "Attr3", "Attr4"]
        
        
        self.THRESHOLD = 0.5

def parse_arguments():
    """Parse command line arguments for training"""
    parser = argparse.ArgumentParser(description='Multi-label Classification Training')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to the dataset folder containing images and label.txt')
    parser.add_argument('--images_folder', type=str, default='images',
                       help='Name of the images folder (default: images)')
    parser.add_argument('--labels_file', type=str, default='label.txt',
                       help='Name of the labels file (default: label.txt)')
    parser.add_argument('--model_name', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet-b0'],
                       help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for multi-label prediction')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create Config object from parsed arguments"""
    config = Config(
        data_path=args.data_path,
        images_folder=args.images_folder,
        labels_file=args.labels_file
    )
    
    
    config.MODEL_NAME = args.model_name
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.THRESHOLD = args.threshold
    
    return config