import random


class ExperienceMemory:

    def __init__(self, capacity, batch_size):
        self.experience = [[]] * capacity
        self.capacity = capacity
        self.batch_size = batch_size
        self.counter = 0

    def add_experience(self, obs, next_obs, reward, action, done):
        self.experience[self.counter] = [obs, next_obs, reward, action, done]
        self.counter = (self.counter + 1) % self.capacity

    def get_experience(self):
        batch_size = self.batch_size
        if self.counter < batch_size:
            batch_size = self.counter

        return random.sample(self.experience, batch_size)
