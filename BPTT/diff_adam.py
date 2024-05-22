import torch
from torch.optim import Adam

a = torch.ones((5,5))
adam = Adam([a], 0.1)
