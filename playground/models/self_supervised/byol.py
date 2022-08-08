"""PyTorch Lightning implementation of BYOL."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union
from copy import deepcopy
from callbacks import BYOLExponentialMovingAverage


class BYOL(pl.LightningModule):
    """PyTorch Lightning re-implementation of Bootstrap Your Own Latent (BYOL).
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

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_dim = input_dim

        self.online_network = None
        self.target_network = deepcopy(self.online_network)
        for p in self.target_network.parameters():
            p.requires_grad = False
            
        self.ema_callback = BYOLExponentialMovingAverage(tau_base=0.996)
    
    def forward(self, x):
        """Returns representation calculate using the online network for x."""
        return self.online_network(x)[0]
    
    def _step(self, v, v_prime):
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
        _, _, h1 = self.online_network(v)
        with torch.no_grad():
            _, z2, _ = self.target_network(v_prime)
            
        return F.cosine_similarity(h1, z2)
        
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
        return [optimizer], [scheduler]