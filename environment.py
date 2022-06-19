from abc import ABC, abstractmethod
import torch
from base import Base
from typing import overload, Callable
from preprocessing import Preprocessing
from my_types import State, Action, Reward, Done
import gym

"""
My wrapper environment

Should support multiple actions
"""

class Environment(Base):

    def __init__(self, device : torch.device, preprocessing: list[Preprocessing] = None, *args, **kwargs):
        if not preprocessing:
            preprocessing = []
        self.preprocessing = preprocessing
        self.device = device
    
    @overload
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        raise NotImplementedError
    
    @abstractmethod
    def size(self) -> tuple[torch.Size, int]:
        raise NotImplementedError
    
    def reset(self) -> None:
        for prep in self.preprocessing:
            prep.ep_reset()
        

class ALE(Environment):

    def __init__(
        self, 
        game_name: str, 
        frameskip: int, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env = gym.make(game_name, frameskip=frameskip, **kwargs)
    
    def step(self, action: Action) -> tuple[State, Reward, Done]:
        # Types are : np.array, float, bool
        next_image, reward, done, _ = self.env.step(action.item())

        for preprocessing in self.preprocessing:
            next_image = preprocessing(next_image)
        
        reward_tensor = torch.tensor([[reward]], device=self.device)
        done_tensor = torch.tensor([[done]], device=self.device)
        return next_image, reward_tensor, done_tensor
    
    def reset(self) -> State:
        super().reset()
        self.env.reset()

        next_image, reward, done, _ = self.env.step(self.env.action_space.sample())
        
        for preprocessing in self.preprocessing:
            next_image = preprocessing(next_image)
        return next_image
        

    def size(self) -> tuple[torch.Size, int]:
        w,h,c = self.env.observation_space.shape
        size = torch.Size([w, h, c])
        for prep in self.preprocessing:
            size = prep.size(size)

        return size, self.env.action_space.n

if __name__ == "__main__":
    import torch
    ale = ALE("ALE/Breakout-v5", [])
    ale.step(torch.tensor([0], dtype=torch.int))
