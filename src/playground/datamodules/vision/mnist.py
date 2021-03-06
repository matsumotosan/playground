import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
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
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False)

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
