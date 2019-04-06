import torch
import torch.nn as nn

scale = 2


# N 1 H W
x = torch.LongTensor(
    [[
        [[1, 1],
         [1, 0]],
        [[2, 1],
         [1, 0]],
        [[2, 1],
         [1, 0]],
    ]]
)

y = torch.LongTensor(
    [[
        [[1, 1],
         [1, 0]],
        [[2, 1],
         [1, 0]],
    ]]
)

print(x.size(), y.size(), torch.cat((x,y), 1))


