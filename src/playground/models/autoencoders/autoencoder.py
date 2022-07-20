import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)


class Autoencoder(pl.LightningModule):
    """PyTorch Lightning re-implementation of vanilla autoencoder.

    Parameters
    ----------
    input_dim :
    
    latent_dim: 
    
    batch_norm:
    
    encoder :
    
    decoder : 
    
    learning_rate:
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        batch_norm: bool = False,
        encoder: Union[nn.Module, None] = None,
        decoder: Union[nn.Module, None] = None,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Set encoder and decoder
        self.encoder = encoder if encoder is not None else resnet18_encoder
        self.decoder = decoder if decoder is not None else resnet18_decoder
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def shared_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z) 
        loss = F.mse_loss(x_hat, x, reduction="mean")
        return loss, {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in log.items()},
            on_step=True,
            on_epoch=False
        )
        return loss, log
    
    def validation_step(self, batch, batch_idx):
        loss, log = self.shared_step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in log.items()})
        return loss, log
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = None
        return [optimizer], [scheduler]