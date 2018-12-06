import torch
import numpy as np

tensor = torch.Tensor([1.0, ])

a = [tensor, tensor2]
print(np.mean([q.detach().numpy() for q in a]))