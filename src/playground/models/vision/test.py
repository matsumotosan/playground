import torch
from resnet import *

resnet18 = _resnet18()
x = torch.randn(2, 3, 224, 224)
y = resnet18(x)
print(y.shape)