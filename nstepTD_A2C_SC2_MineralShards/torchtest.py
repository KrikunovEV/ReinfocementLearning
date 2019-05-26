import numpy as np

rewards = [0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1000, 0, 0, 0, 0]

discounted = []
G = 0
for i in reversed(range(len(rewards))):
    G = rewards[i] + 0.99 * G
    discounted.append(G)

discounted = (discounted - np.mean(discounted)) / np.std(discounted)

print(discounted)

for i in reversed(range(len(rewards))):
    print(discounted[-i-1])

