import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    def __init__(self, numI, numH, numO):

        self.W_IH = np.random.randn(numH, numI) / np.sqrt(numI)
        self.W_HO = np.random.randn(numO, numH) / np.sqrt(numH)

        self.b_IH = np.random.randn(numH) / np.sqrt(numI)
        self.b_HO = np.random.randn(numO) / np.sqrt(numH)

        self.lr = 0.01


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def train(self, inputs, labels):

        # layer 1
        h_data = self.W_IH.dot(inputs) + self.b_IH
        h_data = self.sigmoid(h_data)

        # layer 2
        outputs = self.W_HO.dot(h_data) + self.b_HO
        loss = np.square(outputs - labels).sum()

        # gradients of layer 1
        grad_output = 2 * (outputs - labels)
        grad_w_HO = grad_output.reshape((len(grad_output),1)).dot(h_data.reshape((1,len(h_data))))
        grad_b_HO = grad_output

        # gradients of layer 2
        grad_h = self.W_HO.T.dot(grad_output)
        grad_w_IH = (grad_h * h_data * (1 - h_data)).reshape((len(grad_h),1)).dot(inputs.reshape((1,len(inputs))))
        grad_b_IH = grad_h * h_data * (1 - h_data)

        # optimize
        self.lr *= 0.9999999
        self.W_IH -= self.lr * grad_w_IH
        self.b_IH -= self.lr * grad_b_IH
        self.W_HO -= self.lr * grad_w_HO
        self.b_HO -= self.lr * grad_b_HO

        return loss



    def forwardProp(self, inputs):

        h_data =  self.W_IH.dot(inputs) + self.b_IH
        h_data = self.sigmoid(h_data)

        outputs = self.W_HO.dot(h_data) + self.b_HO
        outputs = self.sigmoid(outputs)

        return outputs


# tradin dataset
inputs = []
labels = []
for i in range(1000):
    x = np.random.random_sample() * 2 - 1
    y = np.random.random_sample() * 2 - 1
    inputs.append([x, y])
    if x > y:
        labels.append([1, 0])
    else:
        labels.append([0, 1])
inputs = np.array(inputs)
labels = np.array(labels)


# training
brain = NeuralNetwork(2, 6, 2)
for iter in range(1000):
    Loss = 0
    for i in range(len(inputs)):
        Loss += brain.train(inputs[i], labels[i])
    print('epoch:',iter, ';loss:', Loss)

# testing
# test dataset
inputs = []
labels = []
for i in range(100):
    x = np.random.random_sample() * 2 - 1
    y = np.random.random_sample() * 2 - 1
    inputs.append([x, y])
    if x > y:
        labels.append([1, 0])
    else:
        labels.append([0, 1])
inputs = np.array(inputs)
labels = np.array(labels)
Accuracy = 0
for i, input in enumerate(inputs):
    predict = np.argmax(brain.forwardProp(input))
    real = np.argmax(labels[i])
    if predict == real:
        Accuracy += 1

print(Accuracy / len(inputs))