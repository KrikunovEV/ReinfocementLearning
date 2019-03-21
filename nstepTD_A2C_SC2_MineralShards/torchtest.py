import torch


x = torch.Tensor([[[1, 2], [1, 2]]]).squeeze_(0)
y = torch.Tensor([[[3, 4], [3, 4]]]).squeeze_(0)
print(torch.cat((x,y), 0))


'''
from pysc2.lib import static_data
depth = max(static_data.UNIT_TYPES) + 1

conv = torch.nn.Conv2d(depth, 1, 1)

x = torch.LongTensor(
    [
        [5, 5, 0, 0],
        [5, 0, 0, 5],
        [5, 0, 5, 0],
        [0, 5, 5, 5]
    ]
)
print(x)

z = torch.unsqueeze(x, 2)
print(z)

y = torch.Tensor(depth, 4, 4).zero_()
y.scatter_(2, z, 1)
print(y)

#conv = torch.nn.Conv2d(depth, 1, 1)
#y = conv(torch.unsqueeze(x, 0))
#print(y)
'''