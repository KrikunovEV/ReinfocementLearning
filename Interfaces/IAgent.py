from abc import ABC, abstractmethod


class IAgent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def action(self, obs):
        pass

    @abstractmethod
    def reward(self, reward):
        pass
