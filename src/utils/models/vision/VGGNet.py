from pyparsing import Forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl


class VGGNet(pl.LightningModule):
    """VGG networks
    
    References
    ----------
    Very deep convolutional networks for large-scale image recognition
    https://arxiv.org/pdf/1409.1556.pdf
    """
    def __init__(self, architecture, hparams, in_channels=3, num_classes=1000):
        super().__init__()
        self.hparams = hparams
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(architecture)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten()
        x = self.fc(x)
        return x

    def configure_optimizers(self, lr):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = 0
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss = 0
        self.log('val_loss', loss)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=(3, 3), 
                        stride=(1, 1), 
                        padding=(1, 1)), 
                    nn.BatchNorm2d(x),
                    nn.ReLU()]
                
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(
                    kernel_size=(2, 2),
                    stride=(2, 2)
                )]
            else:
                pass
        
        return nn.Sequential(*layers)
    

    