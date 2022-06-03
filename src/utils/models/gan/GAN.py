"""PyTorch Lightning implementation of a Vanilla GAN model.
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


def block(in_feat, out_feat, normalize=True):
    """Returns block of linear layer and LeakyReLU activation. 
    Optionally normalize after linear layer."""
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Generator(nn.Module):
    """PyTorch Lightning Vanilla GAN generator."""
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """PyTorch Lightning Vanilla GAN discriminator."""
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid(),   # could be Tanh
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class GAN(pl.LightningModule):
    """PyTorch Lightning Vanilla GAN model."""
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize networks
        data_shape = (channels, width, height)
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim,
            img_shape=data_shape
        )
        self.discriminator = Discriminator(
            img_shape=data_shape
        )

        # Random vector for validation
        self.val_z = torch.randn(8, self.hparams.latent_dim)
        
        # Specify input for 'forward' method
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        
        # Sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim).type_as(imgs)

        # Train generator
        if optimizer_idx == 0:
            # Generate images
            self.generated_imgs = self(z)
        
        # Train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1).type_as(imgs)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []
    
    def on_epoch_end(self) -> None:
        z = self.val_z.type_as(self.generator.model[0].weight)
        
        # Log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated+images", grid, self.current_epoch)