import random

class Experience:
    experience = []

    def addExperience(self, obs, next_obs, reward, action, done):
        self.experience.append([obs, next_obs, reward, action, done])

    def getExperience(self, batch_size):
        if len(self.experience) < batch_size:
            batch_size = len(self.experience)

        return random.sample(self.experience, batch_size)