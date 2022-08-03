import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from typing import Union


class MEME(pl.LightningModule):
    def __init__(
        self,
        input_dim : list[int],
        latent_dim : int,
        encoder = Union[nn.Module, None],
        decoder = Union[nn.Module, None],
        learning_rate : float = 0.0005
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder, self.decoder = get_model('mnist', latent_dim)
    
    def forward(self, x):
        x = self.encoder(x.flatten())
        x = self.decoder(x)
        x = x.view(self.input_dim)
        return x

    def shared_step(self, batch, batch_idx):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999)
        )
        return [optimizer]


def get_model(dataset, latent_dim):
    if dataset == 'mnist':
        encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(True),
            nn.Linear(400, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    elif dataset == 'svhn':
        encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, latent_dim, 4, stride=1, padding=0)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Unflatten(1, (28, 28)),
            nn.Sigmoid()
        )
    elif dataset == 'cub':
        raise NotImplementedError
    elif dataset == 'cub_language':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return encoder, decoder


if __name__ == "__main__":
    input_dim = (1, 28, 28)
    latent_dim = 10
    model = MEME(input_dim, latent_dim)
    
    model