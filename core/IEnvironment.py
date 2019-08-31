from abc import ABC, abstractmethod


class IEnvironment(ABC):

    @abstractmethod
    def reset(self):
        pass  # return observation and reward

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self):
        pass  # return observation and reward


print("Not implemented yet. Leave it.\n")