import math
import pytorch_lightning as pl


class BYOLExponentialMovingAverage(pl.Callback):
    """PyTorch Lightning Callback to update weights of the target network as describe in BYOL.
    
    For the methods in this class to work, the `pl_module` passed must have `online_network` and `target_network` defined.
    """
    def __init__(self, tau_base : float = 0.996):
        super().__init__()
        self.tau_base = tau_base
        self.tau = tau_base
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Update target network weights after """
        self.update_weights(pl_module.online_network, pl_module.target_network)
        self.tau = self.update_tau(pl_module, trainer)
    
    def update_tau(self, pl_module, trainer) -> float:
        """Update tau to be used in next update."""
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.tau_base) * math.cos((math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_network, target_network) -> None:
        """Update weights of target network as a weighted sum of the online network and current target network."""
        for p_online, p_target in zip(online_network.parameters(), target_network.parameters()):
            p_target.data = self.tau * p_target.data + (1 - self.tau) * p_online.data