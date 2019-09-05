from abc import ABC, abstractmethod


class IEnvironment(ABC):
    """Interface describes common behavior of environment."""

    @abstractmethod
    def reset(self):
        """Function has to reset environment to start state.
           Should return environment's state."""
        pass

    @abstractmethod
    def render(self):
        """Function has to render environment's state."""
        pass

    @abstractmethod
    def step(self, action):
        """Function is responsible for transition of environment's state regarding agent's action.
           Should return environment's state."""
        pass
