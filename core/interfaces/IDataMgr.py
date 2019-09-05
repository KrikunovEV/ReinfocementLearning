from abc import ABC, abstractmethod


class IDataMgr(ABC):

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def set(self):
        pass

    @abstractmethod
    def send(self):
        pass


print("Not implemented yet. Leave it.\n")