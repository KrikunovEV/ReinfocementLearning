from abc import ABC, abstractmethod


class IAgent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def reward(self):
        pass


print("Not implemented yet. Leave it.\n")