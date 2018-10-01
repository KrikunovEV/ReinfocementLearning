import torch
import time
import numpy as np


trainData = np.random.uniform(0, 1, (9000, 3))
trainLabel = np.zeros((9000, 3))

for i, data in enumerate(trainData):
    sum = np.sum(np.power(data, 2))

    if sum < 0.25:
        trainLabel[i][0] = 1.0
    elif sum < 0.64:
        trainLabel[i][1] = 1.0
    else:
        trainLabel[i][2] = 1.0



trainData = torch.Tensor(trainData, device="cuda")
trainLabel = torch.Tensor(trainLabel, device="cuda")

model = torch.nn.Sequential(
    torch.nn.Linear(3, 36),
    torch.nn.Tanh(),
    torch.nn.Linear(36, 3),
    torch.nn.Softmax()
)

loss_fn = torch.nn.MSELoss(reduction="sum")

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

start_time = time.time()

for iteration in range(1000):

    #batch = np.random.choice(len(trainData), 256)
    #input = trainData[batch]
    #label = trainLabel[batch]

    pred = model(trainData)

    loss = loss_fn(pred, trainLabel)
    print(iteration, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



testData = np.random.uniform(0, 1, (1000, 3))
testLabel = np.zeros((1000, 3))

for i, data in enumerate(testData):
    sum = np.sum(np.power(data, 2))

    if sum < 0.25:
        testLabel[i][0] = 1.0
    elif sum < 0.64:
        testLabel[i][1] = 1.0
    else:
        testLabel[i][2] = 1.0


testData = torch.Tensor(testData, device="cuda")
Accuracy = 0
for i in range(len(testLabel)):

    pred = model(testData[i])

    if np.argmax(pred.detach().numpy()) == np.argmax(testLabel[i]):
        Accuracy += 1

print("Accuracy is", Accuracy / len(testData))
print("--- %s seconds ---" % (time.time() - start_time))