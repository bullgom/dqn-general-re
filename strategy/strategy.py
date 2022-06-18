from abc import abstractmethod
from serializable import Serializable
from base import Base
from my_types import Q, Action

class Strategy(Base, Serializable):

    @abstractmethod
    def select(self, q: Q) -> Action:
        raise NotImplementedError
