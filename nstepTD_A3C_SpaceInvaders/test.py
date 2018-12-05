import torch

model = torch.nn.Linear(3, 1)

tensor = torch.Tensor([1,2,3])
print(tensor.requires_grad)

tensor = model(tensor)
print(tensor.requires_grad)

tensor = tensor.detach()
print(tensor.requires_grad)