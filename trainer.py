from replay_buffer import ReplayBuffer, Transition
from base import Base
import torch
from my_types import State, Reward, Done
import torch.nn.functional as F

class Trainer(Base):
    """On-Policy is the base trainer"""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        gamma: float,
    ):
        self.buffer = replay_buffer
        self.nn = network
        self.optim = optimizer
        self.bs = batch_size
        self.gamma = gamma

    def step(self) -> torch.Tensor:
        self.steps += 1
        if len(self.buffer) < self.bs:
            return torch.FloatTensor([0])
        
        s_now, a, r, s_next, done = self.buffer.sample(self.bs)

        prediction = self.nn(s_now).gather(0, a)
        target = self.target(r, s_next, done)

        loss = F.smooth_l1_loss(prediction, target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp(-1, 1)
        self.optim.step()

        return loss
    
    def target(self, r:Reward, s_next: State, done: Done) -> torch.Tensor:
        target = r + done * self.gamma * self.nn(s_next).max()
        return target

    def reset(self) -> None:
        self.steps = 0

    def is_full(self) -> bool:
        return len(self.buffer) >= self.bs