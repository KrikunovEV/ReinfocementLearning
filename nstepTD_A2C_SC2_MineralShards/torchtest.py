import torch
import torch.nn as nn
import numpy as np

a = [
    torch.Tensor([2]).cuda(),
    torch.Tensor([3]).cuda(),
    torch.Tensor([4]).cuda()
]

print(np.mean([q.item() for q in a]))

