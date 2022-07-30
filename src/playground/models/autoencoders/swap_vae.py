"""PyTorch Lightning re-implementation of SwapVAE."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union
from pl_bolts.models.autoencoders.components import (
    resnet18_encoder,
    resnet18_decoder
)


class SwapVAE(pl.LightningModule):
    """PyTorch Lightning re-implementation of SwapVAE from 
    'Drop, Swap, and Generate: A Self-Supervised Approach for Generating Neural Activity'
    by Ran Liu, Mehdi Azabou, Max Dabagia, Chi-Heng Lin, Mohammad Gheslaghi Azar, Keith B. Hengen,
    Michal Valko, and Eva Dyer.
    
    https://proceedings.neurips.cc/paper/2021/file/58182b82110146887c02dbd78719e3d5-Paper.pdf
    """
    def __init__(
        self,
        input_dim,
        hidden_dim: list[int],
        content_dim: int,
        style_dim: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        learning_rate: float = 1e-2,
        batchnorm: bool = False,
        encoder: Union[nn.Module, None] = None,
        decoder: Union[nn.Module, None] = None,
        reparameterize_first: bool = True
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.latent_dim = content_dim + style_dim
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.batchnorm = batchnorm
        self.reparameterize_first = reparameterize_first
        
        self.encoder = encoder if encoder is not None else self._get_encoder()
        self.decoder = decoder if decoder is not None else self._get_decoder()
        
        self.fc_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim[-1], self.latent_dim)
    
    def encode(self, x, concatenate=False):
        """Forward pass through encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input vector
        
        concatenate : bool
            If True, returns mu and logvar as a concatenated array.

        Returns
        -------
        mu : torch.Tensor
            Mean
            
        logvar : torch.Tensor
            Log-variance
        """
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        if concatenate:
            return torch.cat((mu, logvar), 0)
        else:
            return mu, logvar
    
    def decode(self, z):
        """Forward pass through decoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent vector
        
        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed input
        """
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x1, x2):
        """Forward pass through SwapVAE architecture.
        
        Parameters
        ----------
        x1 : torch.Tensor
            First view of sample
            
        x2 : torch.Tensor
            Second view of sample
            
        Returns
        -------
        x1_recon : torch.Tensor
            Reconstruction of swapped first view
        
        x2_recon : torch.Tensor
            Reconstruction of swapped second view
        """
        # Encode x1 and x2
        mu1, logvar1 = self.encode(x1)
        mu2, logvar2 = self.encode(x2)
        
        if self.reparameterize_first:
            # Reparameterization trick
            z1 = self.reparameterize(mu1, logvar1)
            z2 = self.reparameterize(mu2, logvar2)
            
            # Perform BlockSwap operation
            z1_new, z2_new = self.swap(z1, z2)
        else:
            pass

        # Decode z1_new and z2_new
        x1_recon = self.decode(z1_new)
        x2_recon = self.decode(z2_new)
        
        return x1_recon, x2_recon
    
    def _step(self, batch, batch_idx):
        x, y = batch
        loss = None
        log = None
        
        # Calculate loss
        loss1, loss_recon1, loss_style1, loss_align1 = self.calculate_loss()
        loss2, loss_recon2, loss_style2, loss_align2 = self.calculate_loss()
        
        logs = {
            
        }
        
        return loss, logs
    
    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        return loss, logs
    
    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        return loss, logs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        
        scheduler = None
        return optimizer, scheduler
    
    def swap(self, z1, z2):
        """Swaps content for two latent vectors.
        
        Parameters
        ----------
        z1 : torch.Tensor
            Latent vector for first view
            
        z2 : torch.Tensor
            Latent vector for second view
            
        Returns
        -------
        z1_new : torch.Tensor
            Content-swapped version of first view
            
        z2_new : torch.Tensor
            Content-swapped version of second view
        """
        # Decouple latent vector into content and style vectors
        c1, s1 = self.decouple(z1)
        c2, s2 = self.decouple(z2)
        
        # Swap content for each latent vector
        z1_new = torch.cat([c2, s1], 0)
        z2_new = torch.cat([c1, s2], 0)
        
        return z1_new, z2_new
    
    def decouple(self, z):
        """Decouples latent vector into content and style vectors.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent vector
        """
        return z[:self.content_dim], z[self.content_dim:]
    
    def calculate_loss(self, distribution='poisson', remove_loss_align=False):
        """Returns sum of reconstruction loss, style space regularization loss, and content space alignment loss.
        
        Parameters
        ----------
        distribution : str
            Distribution model for reconstruction loss
            
        remove_loss_align : bool
            If True, remove alignment term from loss. Otherwise include in loss.
            
        Returns
        -------
        loss : float
            Sum of reconstruction loss, style space regularization loss, and content space alignment loss
        """
        loss_recon = reconstruction_loss(distribution=distribution)
        loss_style = kl_divergence()
        if remove_loss_align:
            pass
        loss_align = alignment_loss()
        loss = loss_recon + self.beta * loss_style + self.alpha * loss_align
        return loss, loss_recon, loss_style, loss_align
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu
    
    def _get_encoder(self):
        dim = [self.input_dim, *self.hidden_dim]
        m = []
        for i in range(len(dim) - 1):
            m.append(nn.Linear(dim[i], dim[i + 1]))
            if self.batchnorm:
                m.append(nn.BatchNorm1d(dim[i + 1]))
            m.append(nn.ReLU(inplace=True))
        return nn.Sequential(*m)

    def _get_decoder(self):
        dim = [self.latent_dim, *self.hidden_dim[::-1], self.input_dim]
        m = []
        for i in range(len(dim) - 1):
            m.append(nn.Linear(dim[i], dim[i + 1]))
            if self.batchnorm:
                m.append(nn.BatchNorm1d(dim[i + 1]))
            m.append(nn.ReLU(inplace=True))
        return nn.Sequential(*m)

def kl_divergence(mu, logvar):
    """Calculates KL divergence for given mean and log-variance."""
    return None


def reconstruction_loss(x, x_recon, distribution='poisson'):
    """Returns reconstruction loss for variey of distributions.
    
    Parameters
    ----------
    x : torch.Tensor
        Original sample
    
    x_recon : torch.Tensor
        Reconstructed sample
    
    distribution : str
        Distribution
        
    Parmaeters
    ----------
    loss : torch.Tensor
        Reconstruction loss based on specified distribution
    """
    if distribution == 'poisson':
        loss = None
    elif distribution == 'gaussian' or distribution == 'l2':
        loss = None
    else:
        raise NotImplementedError
    return loss


def alignment_loss(z1, z2):
    """Returns alignment loss."""
    return None