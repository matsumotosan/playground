"""PyTorch Lightning implementation of BYOL."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Union
from callbacks import BYOLExponentialMovingAverage
from models import resnet18_cifar10


class BYOL(pl.LightningModule):
    """PyTorch Lightning re-implementation of Bootstrap Your Own Latent (BYOL).
    
    Variable names were chosen to match their corresponding parts in the BYOL paper as much as possible.
    """
    def __init__(
        self,
        encoder : Union[str, nn.Module],
        representation_dim : int,
        projector_hidden_dim : int,
        projector_out_dim : int,
        tau_base : float = 0.996,
        **kwargs
    ) -> None:
        super().__init__()

        self.representation_dim = representation_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_out_dim = projector_out_dim

        # Initialize online and target encoders (f)
        self.online_encoder = encoder if isinstance(encoder, nn.Module) else self._get_encoder(encoder)
        self.target_encoder = self._get_target_copy(self.online_encoder)
        
        # Initialize online and target projectors (g)
        self.online_projector = self._get_projector()
        self.target_projector = self._get_target_copy(self.online_projector)
        
        # Initialize online predictor (q)
        self.predictor = self._get_predictor()
        
        self.ema_callback = BYOLExponentialMovingAverage(tau_base=tau_base)
    
    def forward(self, x : Tensor) -> Tensor:
        """Returns representation calculated using the online network for x."""
        y = self.online_encoder(x)
        z = self.online_projector(y)
        return y, z
    
    def _step(self, v : Tensor, v_prime : Tensor) -> Tensor:
        """Calculates loss for online network prediction of target network.
        
        The function was defined following the notation used in the original paper to avoid confusion.
        
        Parameters
        ----------
        v : torch.Tensor
            Online network views
            
        v_prime : torch.Tensor
            Target network views
            
        Returns
        -------
        similarity : torch.Tensor
            Cosine similarity between online network prediction and target network projection
        """
        # Get prediction from online network
        y = self.online_encoder(v)
        z = self.online_projector(y)
        h = self.predictor(z)
        
        # Get projection from target network
        y_prime = self.target_encoder(v_prime)
        z_prime = self.target_projector(y_prime)
        
        # Calculate mean squared error between normalized predictions and target prediction
        loss = -2 * F.cosine_similarity(h, z_prime)
        
        return loss
    
    def shared_step(self, batch, batch_idx):
        """Shared step for training and validation step. 
        
        Computes similarity loss from online network prediction of target network projection in both directions.
        """
        img1, img2 = batch
        
        # Calculate similarity losses in both directions
        loss12 = self._step(img1, img2)
        loss21 = self._step(img2, img1)
        
        return loss12, loss21, loss12 + loss21
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss12, loss21, total_loss = self.shared_step(batch, batch_idx)
        self.log_dict({"loss_12": loss12, "loss_21": loss21, "train_loss": total_loss})
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss12, loss21, total_loss = self.shared_step(batch, batch_idx)
        self.log_dict({"loss_12": loss12, "loss_21": loss21, "val_loss": total_loss})
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update target network weights using EMA update after every training batch."""
        self.ema_callback.on_train_batch_end(
            self.trainer,
            self,
            outputs,
            batch,
            batch_idx
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate, 
        )
        
        scheduler = None
        return [optimizer]

    def _get_encoder(self, encoder):
        if encoder == 'resnet18-cifar':
            # CIFAR-10 variant for ResNet-18 architecutre as described in SimCLR:
            # 1) Replacece first 7x7 conv, stride 2 -> 3x3 conv, stride 1
            # 2) Remove first max pooling operation
            model = resnet18_cifar10()
        elif encoder == 'resnet50':
            model = resnet50()
        else:
            raise NotImplementedError
        return model

    def _get_projector(self):
        return MLP(self.representation_dim, self.projector_hidden_dim, self.projector_out_dim)

    def _get_predictor(self):
        return MLP(self.projector_out_dim, self.projector_hidden_dim, self.projector_out_dim)

    @torch.no_grad()
    def _get_target_copy(self, network):
        return copy.deepcopy(network)


class MLP(nn.Module):
    """MLP class to be used for projector and predictor heads in BYOL."""
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.model(x)