import hydra
import torch
import pytorch_lightning as pl
from models.VGGNet import VGGNet

VGG13 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

@hydra.main(config_path="conf", config_name="config")
def main():
    # Initialize model
    model = VGGNet(architecture=VGG13,
                in_channels=3, 
                num_classes=1000,
                )
    print(model)

    # Initialize DataLoader
    dl = MNISTDataModule()

    # Train model
    trainer = pl.Trainer()
    trainer.fit()

    # Inference


if __name__ == "__main__":
    main()