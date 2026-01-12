import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=25, model_name='efficientnet-b0', pretrained=True):
        super(DiseaseClassifier, self).__init__()
        
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
            
        # Get input feature size of the final layer
        in_features = self.backbone._fc.in_features
        
        # Replace the final fully connected layer
        self.backbone._fc = nn.Identity() # Remove original head
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        # EfficientNet features might be flattened already or not depending on implementation
        # standard efficientnet_pytorch output is the logits from _fc, but we replaced _fc with Identity
        # so features is the pooled output.
        
        return self.classifier(features)
