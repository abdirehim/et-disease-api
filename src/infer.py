import argparse
import sys
import os
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightning_module import DiseaseModule

def load_model(checkpoint_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model = DiseaseModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    return model, config

def preprocess_image(image_path, img_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0) # Add batch dimension

def load_labels(data_path):
    labels_path = Path(data_path) / 'labels.csv'
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        # Invert map to get id -> name
        return dict(zip(df['label_idx'], df['class_name']))
    return None

def predict(model, image_tensor, device, label_map=None):
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    idx = predicted_idx.item()
    conf = confidence.item()
    
    label_name = label_map[idx] if label_map else str(idx)
    
    return label_name, conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='weights/best.ckpt', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    
    try:
        model, config = load_model(args.checkpoint, args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_tensor = preprocess_image(args.image, config['img_size'])
        label_map = load_labels(config['data_path'])
        
        disease, confidence = predict(model, input_tensor, device, label_map)
        
        print(f"Disease: {disease}")
        print(f"Confidence: {confidence:.2%}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained the model and have a valid checkpoint.")
    except Exception as e:
        print(f"An error occurred: {e}")
