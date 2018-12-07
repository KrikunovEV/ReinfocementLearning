import torch
import numpy as np

model = torch.nn.Sequential(
    torch.nn.Linear(2, 2)
)

model2 = torch.nn.Sequential(
    torch.nn.Linear(2, 3)
)

model3 = torch.nn.Sequential(
    torch.nn.Linear(2, 1)
)

tensor = torch.Tensor([1.0, 5.0])

tensor = model(tensor)
loss1 = model2(tensor).sum()
loss2 = model3(tensor).sum()

(loss1 + loss2).backward()

for param in model.parameters():
    print(param.grad)
    print(param._grad)
    print()