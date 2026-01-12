import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .efficientnet import DiseaseClassifier

class DiseaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.model = DiseaseClassifier(
            num_classes=config['num_classes'],
            model_name=config['model_name'],
            pretrained=config['pretrained']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                patience=3, 
                factor=0.5
            ),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
