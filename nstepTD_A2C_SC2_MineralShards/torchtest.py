import torch
import torch.nn as nn
import numpy as np



a = torch.Tensor([5, 6])
b = torch.Tensor([1, 1 , 1, 1,1,1,1,1])
print(torch.cat((a, b)).var())

