import torch
import torch.nn as nn
import torchvision.models as models

class MultiLabelModel(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=4, pretrained=True):
        super(MultiLabelModel, self).__init__()
        
        # Load pretrained backbone
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_features = self.backbone.fc.in_features  # This is 2048
            self.backbone.fc = nn.Identity()
            
            # Add a projection layer to reduce 2048 -> 512 to match your trained model
            self.projection = nn.Linear(backbone_features, 512)
            
        elif model_name == 'efficientnet-b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Classification head for multi-label (expects 512 input features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 512),  # Input 512, output 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Output num_classes (4)
        )
        
    def forward(self, x):
        features = self.backbone(x)  # This gives 2048 features for ResNet50
        if hasattr(self, 'projection'):
            features = self.projection(features)  # Reduce to 512 features
        output = self.classifier(features)
        return output