from .strategy import Strategy
from my_types import Q, Action
import torch


class EpsilonGreedy(Strategy):
    def __init__(self, start: float, end: float, decay_steps: int):
        """
        Select randomly with epsilon probability
        Linearly decreases from `start` to `end` over `steps` steps
        """
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

        self.slope = (end-start)/decay_steps

    def select(self, q: Q) -> Action:
        """
        If multiple action dimension, run this for each dimension
        """
        size = q.size()
        batch_size = size[0]
        action_count = size[-1]
        mask = torch.rand(batch_size) <= self.epsilon

        a = q.argmax(dim=1)
        a[mask] = torch.randint(0, action_count, (batch_size,))

        return a

    def reset(self) -> None:
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1

    @property
    def epsilon(self) -> float:
        return max(self.start + self.slope * self.current_step, self.end)

    def serialize(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "decay_steps": self.decay_steps
        }
