from abc import ABC, abstractmethod


class Base(ABC):
    """
    What name should I give this?
    Every thing should be subclass of this
    """

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
