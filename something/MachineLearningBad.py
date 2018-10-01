import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Point:

    def __init__(self):
        self.x = np.random.random_sample() * 2 - 1
        self.y = np.random.random_sample() * 2 - 1
        if self.x**2 + self.y**2 <= 0.25:
            self.label = 1
        else:
            self.label = -1


class Perceptron():

    def __init__(self, n):
        self.W = np.random.randn(n) * 0.001

    def ActivationFunctionSign(self, data):
        return np.sign(data)

    def predict(self, inputs):
        data = self.W.dot(inputs)
        output = self.ActivationFunctionSign(data)
        return output

    def train(self, data, label):
        prediction = self.predict(data)
        loss = label - prediction
        self.W += data * loss * 0.01


train_size = 100
points = [Point() for i in range(train_size)]

brain = Perceptron(3)
for p in points:
    brain.train(np.array([1, p.x, p.y]), p.label)

for p in points:
    if brain.predict(np.array([1, p.x, p.y])) == p.label:
        plt.plot(p.x, p.y, color='black', marker='o', markersize='10')
        plt.plot(p.x, p.y, color='green', marker='o', markersize='7')
    else:
        plt.plot(p.x, p.y, color='black', marker='o', markersize='10')
        plt.plot(p.x, p.y, color='red', marker='o', markersize='7')



ax = plt.gcf().gca()
ax.add_artist(Circle((0, 0), 0.5, color='green', fill=False))
plt.plot([-1, 1], [(-brain.W[0] + brain.W[1]) / brain.W[2], (-brain.W[0] - brain.W[1]) / brain.W[2]], color='red')
plt.show()

