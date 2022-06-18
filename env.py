from abc import ABC, abstractmethod
import torch
from base import Base
from typing import overload, Callable
from my_types import State, Action, Reward, Done
import gym

"""
My wrapper environment

Should support multiple actions
"""

class Environment(Base):

    def __init__(self, preprocessing: list[Callable]):
        self.preprocessing = preprocessing
    
    @overload
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        raise NotImplementedError

class ALE(Environment):

    def __init__(self, game_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.env = gym.make(game_name)
    
    def step(self, action: Action) -> tuple[State, Reward, Done]:
        # Types are : np.array, float, bool
        next_image, reward, done, _ = self.env.step(action.item())
        return next_image, reward, done
    
    def reset(self) -> None:
        pass

if __name__ == "__main__":
    import torch
    ale = ALE("ALE/Breakout-v5", [])
    ale.step(torch.tensor([0], dtype=torch.int))
