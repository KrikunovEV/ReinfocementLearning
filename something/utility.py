import numpy as np


def GetData(amountData):

    #spheres

    # 3 classes:
    # the first - into sphere with radius 0.5
    # the second - onto donut between 0.5 - 1.0 radiuses
    # the third - out sphere with radius 1.0

    nTrain = int(amountData * 0.8)
    nTest = amountData - nTrain

    # TRAIN DATASET
    trainData = np.empty(shape=(nTrain, 3))
    trainLabel = np.empty(shape=(nTrain, 3))
    for i in range(nTrain):

        # NORMALIZE COORDINATE [-1; 1] + ZERO CENTERED
        coors = np.random.random_sample(3) * 2 - 1
        trainData[i] = coors

        norm = np.sqrt(np.sum(coors**2))
        if norm < 0.5:
            trainLabel[i] = np.array([1, 0, 0])
        elif norm < 1.0:
            trainLabel[i] = np.array([0, 1, 0])
        else:
            trainLabel[i] = np.array([0, 0, 1])

    # TEST DATASET
    testData = np.empty(shape=(nTest, 3))
    testLabel = np.empty(shape=(nTest, 3))
    for i in range(nTest):

        # NORMALIZE COORDINATE [-1; 1] + ZERO CENTERED
        coors = np.random.random_sample(3) * 2 - 1
        testData[i] = coors

        norm = np.sqrt(np.sum(coors ** 2))
        if norm < 0.5:
            testLabel[i] = np.array([1, 0, 0])
        elif norm < 1.0:
            testLabel[i] = np.array([0, 1, 0])
        else:
            testLabel[i] = np.array([0, 0, 1])


    return trainData, trainLabel, testData, testLabel
