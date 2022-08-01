import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class SyntheticReachingDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]
            )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data = None
            self.train_data, self.val_data = random_split(data, [55000, 5000])
        if stage == "test" or stage is None:
            self.test_data = None

    def train_dataloader(self):
        train_dl = DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return test_dl


def generate_data(dim=2, a=5.0, b=0.3, l=4):
    """Generate synthetic reaching dataset as described in SwapVAE by Dyer et al. (2021).
    
    Parameters
    ----------
    dim : int
        Data dimensionality
    
    a : float
        Scaling factor for Gaussians
        
    b : float
        Scaling factor for variance
        
    l : int
        Number of data points sampled within each cluster
        
    Returns
    -------
    pts : array of shape ()
        Generated points
    """
    
    return None


def realnvp():
    return None