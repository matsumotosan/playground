"""PyTorch Lightning re-implementation of VAE."""
from os import stat
import pytorch_lightning as pl
import torch
from torch import nn
from types import Union
from pl_bolts.models.autoencoders.components import (
    resnet18_encoder,
    resnet18_decoder
)


class VanillaVAE(pl.LightningModule):
    """PyTorch Lightning re-implementation of a vanilla variational autoencoder (VAE).

    Parameters
    ----------

    References
    ----------
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dim: int,
        learning_rate: float = 1e-2,
        batchnorm: bool = False,
        encoder: Union[nn.Module, None] = None,
        decoder: Union[nn.Module, None] = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batchnorm = batchnorm

        # Set encoder and decoder
        self.encoder = encoder if encoder is not None else self._get_encoder(hidden_dim)
        self.decoder = decoder if decoder is not None else self._get_decoder(hidden_dim)

        # Fully connected lauers for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim[-1], self.latent_dim)
        self.fv_log_var = nn.Linear(hidden_dim[-1], self.latent_dim)

    @staticmethod
    def _get_encoder(dims: list[int]):
        pass
    
    @staticmethod
    def _get_decoder(dims: list[int]):
        pass

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def encoder(self, x):
        pass
    
    def decode(self, x):
        pass
    
    def shared_step(self, batch, batch_idx):
        x, _ = batch

        # Encode input
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # Sample latent vector (z) from prior distribution (q)


        # Decode
        x_hat = self.decoder(z)

        # Calculate reconstruction loss based on chosen prior distribution
        loss = self.gaussian_likelihood()

        # Calculate KL divergence
        kl_div = self.kl_divergence()

        # Calculate evidence lower bound (ELBO)
        elbo = (kl_div - loss).mean()

        return elbo

    def training_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.shared_step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

        scheduler = 0
        return [optimizer], [scheduler]
    
    def sample(self, n):
        z = torch.randn(n, self.latent_dim)
        samples = self.decode(z)
        return samples
        
    def reconstruct(self, mu, log_var):
        pass