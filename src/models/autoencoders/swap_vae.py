"""PyTorch Lightning re-implementation of SwapVAE."""
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from types import Union
from pl_bolts.models.autoencoders.components import (
    resnet18_encoder,
    resnet18_decoder
)


class SwapVAE(pl.LightningModule):
    """PyTorch Lightning re-implementation of SwapVAE.
    
    Parameters
    ----------
    encoder : 
        Encoder architecutre
    
    decoder : 
        Decoder architecture
    
    References
    ----------
    """
    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dim: list = [163, 128],
        learning_rate: float = 1e-2,
        batchnorm: bool = False,
        encoder: Union[nn.Module, None] = None,
        decoder: Union[nn.Module, None] = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize encoder and decoder
        self.encoder = encoder if encoder is not None else resnet18_encoder
        self.decoder = decoder if decoder is not None else resnet18_decoder
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def shared_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate)
        
        scheduler = 0
        return [optimizer], [scheduler]