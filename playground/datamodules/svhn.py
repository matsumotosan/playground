import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import SVHN


class SVHNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        transform=None,
        split: float =0.8
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.num_classes = 10
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081))
            ])

    def prepare_data(self) -> None:
        """Download SVHN data."""
        SVHN(self.data_dir, split='train', download=True, transform=self.transform)
        SVHN(self.data_dir, split='test', download=True, transform=self.transform)

    def setup(self, stage=None) -> None:
        """Assign training, validation, and test data splits."""
        if stage == "fit" or stage is None:
            svhn_full = SVHN(self.data_dir, split='train', transform=self.transform)
            n_train = int(self.split * len(svhn_full))
            n_val = len(svhn_full) - n_train
            self.svhn_train, self.svhn_val = random_split(svhn_full, [n_train, n_val])
        if stage == "test" or stage is None:
            self.svhn_test = SVHN(self.data_dir, split='test', transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(self.svhn_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(self.svhn_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return testing DataLoader."""
        return DataLoader(self.svhn_test, batch_size=self.batch_size, num_workers=self.num_workers)
