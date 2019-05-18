import numpy as np

rewards = [1, 0, 500, 0, 9, 10, 1000]

discounted = []
G = 0
for i in reversed(range(len(rewards))):
    G = rewards[i] + 0.99 * G
    discounted.append(G)

print(np.vstack(discounted))

discounted = (discounted - np.mean(discounted)) / np.std(discounted)

print()
print(np.vstack(discounted))