import torch
import torch.nn as nn
import pytorch_lightning as pl


class SimCLR(pl.LightningDataModule):
    def __init__(self):
        self.save_hyperparameters()
        
    
    def forward(self):
        pass
    
    def training_step(self):
        pass
    
    def validation_step(self):
        pass
    
    def configure_optimizers(self):
        pass
    
    
def nt_xent_loss(out1, out2, temperature):
    out = torch.cat([out1, out2], dim=0)
    n_samples = len(out)
    
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)