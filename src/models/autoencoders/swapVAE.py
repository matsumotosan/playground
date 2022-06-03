from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl


class SwapVAE(pl.LightningModule):
    """PyTorch Lightning implementation of SwapVAE.
    """
    def __init__(
        self,
        num_classes: int
    ):
        super().__init__()
    
    def forward(self):
        pass

    def training_step(self):
        pass
    
    def configure_optimizers(self):
        pass