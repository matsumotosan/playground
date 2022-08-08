from .cifar10 import CIFAR10DataModule
from .mnist import MNISTDataModule
from .svhn import SVHNDataModule


__all__ = [
    "CIFAR10DataModule",
    "MNISTDataModule",
    "SVHNDataModule"
]
