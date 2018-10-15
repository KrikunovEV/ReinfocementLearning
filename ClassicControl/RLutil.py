import random
import torch
import numpy as np

class Experience:

    myExperience = []
    BATCH_SIZE = 16

    def addNewExperience(self, obs, reward, action, next_obs, terminal):
        self.myExperience.append([obs, reward, action, next_obs, terminal])

    def replayExperience(self, model, loss_fn, optimizer, DISCOUNT_FACTOR):

        batch_size = self.BATCH_SIZE
        if len(self.myExperience) < self.BATCH_SIZE:
            batch_size = len(self.myExperience)

        batch = random.sample(self.myExperience, batch_size)
        for obs, reward, action, next_obs, terminal in batch:

            q_new = reward
            if not terminal:
                q_new += DISCOUNT_FACTOR * np.max(model.forward(next_obs).detach().numpy())

            q_values = model.forward(obs)
            q_values[action] = q_new

            q_predicted = model(obs)

            loss = loss_fn(q_predicted, q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



