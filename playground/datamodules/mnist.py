import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        transform=None,
        train_split: float = 0.8
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081))
            ])

    def prepare_data(self) -> None:
        """Download MNIST data."""
        MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage=None) -> None:
        """Assign training, validation, and test data splits."""
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            n_train = int(self.train_split * len(mnist_full))
            n_val = len(mnist_full) - n_train
            self.mnist_train, self.mnist_val = random_split(mnist_full, [n_train, n_val])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    @property
    def num_classes(self) -> int:
        return 10

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return testing DataLoader."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
