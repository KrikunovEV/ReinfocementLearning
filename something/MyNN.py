import numpy as np


class NeuralNetwork():

    def __init__(self, numI, numO):

        self.numI = numI
        self.numO = numO
        self.lr = 0.05
        self.lamb = 0.001

        self.W = []
        self.bias = []
        self.numLayers = [numI]


    def AddFCLayer(self, numH):
        self.numLayers.append(numH)


    def Initialization(self):
        self.numLayers.append(self.numO)

        if len(self.numLayers) == 2:
            w = np.random.randn(self.numO, self.numI) * 0.01
            b = np.random.randn(self.numO) * 0.01
            self.W.append(w)
            self.bias.append(b)
        else:
            for i in range(1, len(self.numLayers) - 1):
                numI, numH, numO = self.numLayers[i-1], self.numLayers[i], self.numLayers[i+1]
                w_IH = np.random.randn(numH, numI) * 0.01
                w_HO = np.random.randn(numO, numH) * 0.01
                b_IH = np.random.randn(numH) * 0.01
                b_HO = np.random.randn(numO) * 0.01
                self.W.append(w_IH)
                self.W.append(w_HO)
                self.bias.append(b_IH)
                self.bias.append(b_HO)


    def Sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def tanh(self, x):
        return np.tanh(x)


    def Loss(self, x, label):
        return np.square(x - label).sum()


    def Softmax(self, x):
        e = np.exp(x)
        return e / np.sum(e)

    def SoftmaxLoss(self, x, label):
        y = np.log(self.Softmax(x))
        return - np.sum(y * label) + np.sum(self.W[-1]**2) / 2 * self.lamb


    def InitializeAndTrain(self, inputs, labels):
        self.Initialization()

        Loss = []

        for iteration in range(1, 5001):
            loss = 0

            batch = np.random.choice(len(inputs), 256)
            batch_inputs = inputs[batch]
            batch_labels = labels[batch]

            for input, label in zip(batch_inputs, batch_labels):

                # FORWARD
                layers_data = [input]
                for i, (w, b) in enumerate(zip(self.W, self.bias)):
                    h = w.dot(layers_data[-1]) + b
                    if i != len(self.W) - 1:
                        h = self.tanh(h)
                    else:
                        h = self.Softmax(h)
                    layers_data.append(h)
                loss += self.SoftmaxLoss(layers_data[-1], label)

                # BACKWARD
                grad = []
                grad_w = []
                grad_b = []
                for i in range(len(self.W) - 1, -1, -1):
                    if i == len(self.W) - 1:
                        grad.append(layers_data[-1] - label) # de/dy * dy/dz
                        grad_w.append(grad[-1].reshape((len(grad[-1]), 1)).dot
                                      (layers_data[-2].reshape((1, len(layers_data[-2])))) + self.lamb * self.W[-1])
                        grad_b.append(grad[-1])
                    else:
                        grad.append(self.W[i+1].T.dot(grad[-1]))
                        grad_b.append(grad[-1] * (1 - layers_data[i + 1] ** 2))
                        grad_w.append(grad_b[-1].reshape((len(grad[-1]), 1)).dot
                                      (layers_data[i].reshape((1, len(layers_data[i])))))

                # OPTIMIZE
                for i in range(len(self.W) - 1, -1, -1):
                    j = len(self.W) - (i + 1)
                    self.W[i] -= grad_w[j] * self.lr
                    self.bias[i] -= grad_b[j] * self.lr

            if iteration % 500 == 0:
                self.lr *= 0.9
            Loss.append(loss)
            print('epoch:', iteration, ';  loss:', loss)

        return Loss


    def Forward(self, input, label):

        h_data = input
        for i, (w, b) in enumerate(zip(self.W, self.bias)):
            h_data = w.dot(h_data) + b
            if i != len(self.W) - 1:
                h_data = self.tanh(h_data)
            else:
                h_data = self.Softmax(h_data)

        return h_data, self.SoftmaxLoss(h_data, label)