import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):

    # 1 layer
    h = 1 / (1 + np.exp(-x.dot(w1)))
    # 2 layer
    y_pred = h.dot(w2)
    # loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # find gradient (backpropagation) and optimize
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2

print(w2)

