from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        pass
    
    def train(self: T, mode: bool = True) -> T:
        return super().train(mode)
    
    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam()
        return optimizer
