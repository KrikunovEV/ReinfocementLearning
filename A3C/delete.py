import gym
import matplotlib.pyplot as plt
import numpy as np

def Preprocess(img):
    img = img[::2, ::2]
    img = img[10:len(img)-7]
    return np.mean(img, axis=2)[:,:].astype(np.float32) / 255.0

env = gym.make('Breakout-v0')
print(env.action_space)
plt.imshow(env.reset(), cmap='gray')
plt.show()



obs = Preprocess(env.reset())

plt.imshow(obs, cmap='gray')
plt.show()