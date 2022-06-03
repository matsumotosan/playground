import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl


class AlexNet(pl.LightningModule):
    """AlexNet architecture
    
    References
    ----------
    """
    def __init__(self, in_channels=3) -> None:
        super.__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=3, 
                out_channels=96, 
                kernel_size=11, 
                stride=4
            ),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv2d(),
            nn.ReLU(),
            
            # Layer 4
            nn.Conv2d(),
            nn.ReLU()
        )
        self.init_weights()
    
    def init_weights(self) -> None:
        pass
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss = F.cross_entropy(output, target)
        return loss
    
    def eval_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss = F.cross_entropy(output, target)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')