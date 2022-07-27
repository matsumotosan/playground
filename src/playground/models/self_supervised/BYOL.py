"""PyTorch Lightning implementation of BYOL."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union
from copy import deepcopy


class BYOL(pl.LightningModule):
    """PyTorch Lightning implementation of Bootstrap Your Own Latent (BYOL).
    """
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        input_dim: int,
        batch_size: int,
        base_encoder: Union[str, nn.Module],
        encoder_out_dim: int,
        projector_hidden_size: int,
        projector_out_dim: int,
        **kwargs
    ):
        super().__init__()
        if 'online_network' in kwargs:
            self.online_network = kwargs.get('online_network')
        else:
            self.online_network = None

        self.target_network = deepcopy(self.online_network)
    
    def forward(self, x):
        pass
    
    def shared_step(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = 0
        scheduler = 0
        return [optimizer], [scheduler]
    
    