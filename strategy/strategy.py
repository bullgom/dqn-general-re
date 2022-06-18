from abc import abstractmethod
import torch

from serializable import Serializable


class Strategy(Serializable):

    @abstractmethod
    def select(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError
