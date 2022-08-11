import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from callbacks import BYOLExponentialMovingAverage


class MYOW(pl.LightningModule):
    def __init__(
        self,
        lam : float = 0.1,
        k : int = 1,
        L : int = 512,
        learning_rate : float = 0.03
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        
        self.online_network = None
        self.target_network = None
        self.ema_callback = BYOLExponentialMovingAverage()
    
    def forward(self):
        pass
    
    def training_step(self):
        pass
    
    def validation_step(self):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return [optimizer]


if __name__ == "__main__":
    model = MYOW()
    x = torch.randn(100)
    output = model(x)
    output.shape