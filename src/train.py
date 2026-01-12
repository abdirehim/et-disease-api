import os
import argparse
import yaml
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
from pathlib import Path

# Add project root to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightning_module import DiseaseModule

class DiseaseDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        # Walk through directory
        for cls_dir in self.root_dir.iterdir():
            if cls_dir.is_dir():
                label_name = cls_dir.name
                # We need to load labels map if we want consistent labels, 
                # but for simplicity we can assume folder names are enough and use what was generated
                # Actually, better to load labels.csv to ensure mapping is consistent with inference
                pass
                
        # To ensure we match the labels.csv mapping, let's load it
        labels_path = Path(root_dir) / 'labels.csv'
        if labels_path.exists():
            self.label_df = pd.read_csv(labels_path)
            self.label_map = dict(zip(self.label_df['class_name'], self.label_df['label_idx']))
        else:
             # Fallback if preprocessing wasn't perfectly consistent via scripts
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.label_map = {c: i for i, c in enumerate(classes)}
            
        for cls_name, label_idx in self.label_map.items():
            cls_dir = self.root_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob('*'):
                    self.samples.append((str(img_path), label_idx))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default to tensor conversion
            image = A.Compose([A.Normalize(), ToTensorV2()])(image=image)['image']
            
        return image, label

def get_transforms(config, split='train'):
    if split == 'train' and config['augment']:
        return A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.HorizontalFlip(p=config['horizontal_flip_prob']),
            A.VerticalFlip(p=config['vertical_flip_prob']),
            A.Rotate(limit=config['rotate_limit'], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def train(config):
    pl.seed_everything(42)
    
    # Data
    train_dataset = DiseaseDataset(config['data_path'], 'train', transform=get_transforms(config, 'train'))
    val_dataset = DiseaseDataset(config['data_path'], 'val', transform=get_transforms(config, 'val'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Model
    model = DiseaseModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        mode='min'
    )
    
    logger = TensorBoardLogger(config['log_dir'], name='disease_classification')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision=config['precision']
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train(config)
