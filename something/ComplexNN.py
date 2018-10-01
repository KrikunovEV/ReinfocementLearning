from utility import *
from MyNN import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


trainData, trainLabel, testData, testLabel = GetData(5000)


brain = NeuralNetwork(3, 3)
brain.AddFCLayer(36)
Loss = brain.InitializeAndTrain(trainData, trainLabel)


Accuracy = 0
LossF = 0
for input, label in zip(testData, testLabel):
    predict, loss = brain.Forward(input, label)
    predict = predict.argmax()
    LossF += loss
    if np.argmax(label) == predict:
        Accuracy += 1
print('Accuracy =', Accuracy / len(testData), 'Loss:', LossF)


plt.plot(np.arange(len(Loss)), Loss)
plt.show()


# SUPER-PUPER VISUALIZER OF DATA
# Waste of time :C
'''
fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

circles = [Circle((0, 0), 0.5, color='r', fill=False) for i in range(3)]
ax1.add_patch(circles[0])
ax1.add_patch(circles[1])
ax1.add_patch(circles[2])
art3d.pathpatch_2d_to_3d(circles[0])
art3d.pathpatch_2d_to_3d(circles[1], zdir='x', z=0)
art3d.pathpatch_2d_to_3d(circles[2], zdir='y', z=0)

circles1 = [Circle((0, 0), 0.5, color='r', fill=False) for i in range(3)]
circles2 = [Circle((0, 0), 0.8, color='g', fill=False) for i in range(3)]
ax2.add_patch(circles1[0])
ax2.add_patch(circles1[1])
ax2.add_patch(circles1[2])
ax2.add_patch(circles2[0])
ax2.add_patch(circles2[1])
ax2.add_patch(circles2[2])
art3d.pathpatch_2d_to_3d(circles1[0])
art3d.pathpatch_2d_to_3d(circles1[1], zdir='x', z=0)
art3d.pathpatch_2d_to_3d(circles1[2], zdir='y', z=0)
art3d.pathpatch_2d_to_3d(circles2[0])
art3d.pathpatch_2d_to_3d(circles2[1], zdir='x', z=0)
art3d.pathpatch_2d_to_3d(circles2[2], zdir='y', z=0)

circles = [Circle((0, 0), 0.8, color='b', fill=False) for i in range(3)]
ax3.add_patch(circles[0])
ax3.add_patch(circles[1])
ax3.add_patch(circles[2])
art3d.pathpatch_2d_to_3d(circles[0])
art3d.pathpatch_2d_to_3d(circles[1], zdir='x', z=0)
art3d.pathpatch_2d_to_3d(circles[2], zdir='y', z=0)

ax1.scatter(trainData[:, 0][trainLabel[:, 0] == 1], trainData[:, 1][trainLabel[:, 0] == 1],
           trainData[:, 2][trainLabel[:, 0] == 1], color='r')
ax2.scatter(trainData[:, 0][trainLabel[:, 1] == 1], trainData[:, 1][trainLabel[:, 1] == 1],
           trainData[:, 2][trainLabel[:, 1] == 1], color='g')
ax3.scatter(trainData[:, 0][trainLabel[:, 2] == 1], trainData[:, 1][trainLabel[:, 2] == 1],
           trainData[:, 2][trainLabel[:, 2] == 1], color='b')

ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1, 1)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)
ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)
ax3.set_zlim(-1, 1)

plt.show()
'''