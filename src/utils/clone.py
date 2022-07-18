import copy
import torch.nn as nn


def clone_module(module, N):
    """Returns a ModuleList of N identical modules."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
