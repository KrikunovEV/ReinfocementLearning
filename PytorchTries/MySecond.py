import torch
import time
import numpy as np
from sklearn.datasets import fetch_mldata


mnist = fetch_mldata('MNIST original', data_home="./")
all_data = mnist.data.reshape((-1, 1, 28, 28))
all_label = mnist.target

k = 0
for i in range(1, len(all_label)):
    if all_label[i-1] == 9.0 and all_label[i] == 0.0:
        k = i
        break
print(k)


trainData = torch.Tensor(all_data[:k]).cuda()
trainLabel = torch.LongTensor(all_label[:k]).cuda()
testData = torch.Tensor(all_data[k:]).cuda()
testLabel = torch.LongTensor(all_label[k:]).cuda()

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 10, 5),
    torch.nn.MaxPool2d(kernel_size=2),
).cuda()

model2 = torch.nn.Sequential(
    torch.nn.Linear(12*12*10, 10),
    torch.nn.ReLU()
).cuda()

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)

start_time = time.time()

for iteration in range(50):

    batch = np.random.choice(len(trainData), 256)
    input = trainData[batch]
    label = trainLabel[batch]

    pred = model(input)
    pred = pred.view(-1, 12*12*10)
    pred = model2(pred)

    loss = loss_fn(pred, label)
    print(iteration, loss.item())

    optimizer.zero_grad()
    optimizer2.zero_grad()

    loss.backward()

    optimizer.step()
    optimizer2.step()



Accuracy = 0

pred = model(testData)
pred = pred.view(-1, 12 * 12 * 10)
pred = model2(pred)

for i, p in enumerate(pred):
    if np.argmax(p.cpu().detach().numpy()) == np.argmax(testLabel[i]):
        Accuracy += 1

print("Accuracy is", Accuracy / len(testData))
print("--- %s seconds ---" % (time.time() - start_time))
