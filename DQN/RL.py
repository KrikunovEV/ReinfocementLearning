import numpy as np
import torch

def getEpsilon(epsilon, threshold):
    if epsilon > threshold:
        return epsilon - 0.000001
    return threshold

def preprocess(img):
    img = img[::2, ::2]
    return np.mean(img, axis=2)[np.newaxis,:,:].astype(np.float32) / 255.0

def getReward(reward):
    return np.sign(reward)

def getQvalues(model, data, gamma):
    obs = torch.Tensor([data[i][0] for i in range(len(data))]).cuda()
    next_obs = torch.Tensor([data[i][1] for i in range(len(data))]).cuda()
    reward = torch.Tensor([data[i][2] for i in range(len(data))]).cuda()
    action = [data[i][3] for i in range(len(data))]
    done = torch.ByteTensor([data[i][4] for i in range(len(data))]).cuda()

    maxQvalues = gamma * torch.max(model.noGradForward(next_obs), 1)[0]
    Qnew = torch.where(done, reward, maxQvalues)

    Qvalues = model.noGradForward(obs)
    for i in range(len(action)):
        Qvalues[i][action[i]] = Qnew[i]

    return Qvalues