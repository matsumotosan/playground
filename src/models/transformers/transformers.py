"""PyTorch Lightning re-implementation of a Transformer."""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from types import Union
from pl_bolts.models.autoencoders.components import (
    resnet18_encoder,
    resnet18_decoder
)


class Transformer(pl.LightningModule):
    def __init__(self):
        pass
    
    def configure_optimizers(self):
        pass
    
    def forward(self):
        pass
    
    def training_step(self):
        pass
    
    def validation_step(self):
        pass
    
    def shared_step(self):
        pass