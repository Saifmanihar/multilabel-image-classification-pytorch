import torch
from PIL import Image
import argparse
from torchvision import transforms
import os

from config import Config
from models import MultiLabelModel

def parse_inference_args():
    """Parse command line arguments for inference"""
    parser = argparse.ArgumentParser(description='Multi-label Classification Inference')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset (if model was trained with specific path)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for predictions')
    parser.add_argument('--model_name', type=str, default='resnet50',
                       help='Model architecture used for training')
    return parser.parse_args()

def load_model(model_path, model_name, num_classes=4, device='cpu'):
    """Load the trained model"""
    # Create model
    model = MultiLabelModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load the state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model.to(device)

def preprocess_image(image_path, img_size=224, mean=None, std=None):
    """Preprocess image for inference"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(image_path, model, config):
    """Make prediction for a single image"""
    # Preprocess image
    image_tensor = preprocess_image(
        image_path, 
        config.IMG_SIZE, 
        config.MEAN, 
        config.STD
    )
    
    if image_tensor is None:
        return None
    
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Get predictions based on threshold
    predictions = (probabilities > config.THRESHOLD).astype(int)
    
    return predictions, probabilities

def main():
    args = parse_inference_args()
    
    
    config = Config(data_path=args.data_path)
    config.THRESHOLD = args.threshold
    config.MODEL_NAME = args.model_name
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found!")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found!")
        print("Please train the model first using train.py")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(
        args.model, 
        args.model_name, 
        num_classes=config.NUM_CLASSES,
        device=device
    )
    
    # Make prediction
    print(f"Predicting for image: {args.image}")
    predictions, probabilities = predict(args.image, model, config)
    
    if predictions is not None:
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        # Print attributes present
        present_attrs = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            attr_name = config.CLASS_NAMES[i]
            status = "PRESENT" if pred == 1 else "ABSENT"
            print(f"{attr_name}: {status} (confidence: {prob:.3f})")
            
            if pred == 1:
                present_attrs.append(attr_name)
        
        print("\n" + "="*60)
        print(f"Attributes present: {', '.join(present_attrs) if present_attrs else 'None'}")
        print("="*60)
    else:
        print("Prediction failed!")

if __name__ == "__main__":
    main()