import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform=None, split=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.num_classes = 10
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081))
            ])

    def prepare_data(self) -> None:
        """Download MNIST data."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Assign training, validation, and test data splits."""
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [int(self.split * len(mnist_full)), len(mnist_full) - int(self.split * len(mnist_full))]
            )
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        """Returns training DataLoader."""
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Returns validation DataLoader."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Returns testing DataLoader."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
