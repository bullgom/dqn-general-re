from replay_buffer import ReplayBuffer, Transition
from architecture import Network
from base import Base
import torch
from my_types import State, Reward, Done
import torch.nn.functional as F

class Trainer(Base):
    """On-Policy is the base trainer"""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        network: Network,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        gamma: float,
        steps_per_update: int
    ):
        self.buffer = replay_buffer
        self.nn = network
        self.optim = optimizer
        self.bs = batch_size
        self.gamma = gamma
        self.steps_per_update = steps_per_update

    def step(self) -> torch.Tensor:
        loss = self.train()
        self.steps += 1
        return loss
    
    def train(self) -> torch.FloatTensor:
        if self.steps % self.steps_per_update != 0:
            return torch.FloatTensor([0])

        if len(self.buffer) < self.bs:
            return torch.FloatTensor([0])
        
        s_now, a, r, s_next, done = self.buffer.sample(self.bs)

        prediction_temp = self.nn(s_now).detach()
        prediction = prediction_temp.gather(1, a)
        target = self.target(r, s_next, done)

        loss = F.smooth_l1_loss(prediction, target)

        self.optim.zero_grad()
        loss.backward()
        for param in self.nn.parameters():
            param.grad.data.clamp(-1, 1)
        self.optim.step()

        return loss
    
    def target(self, r:Reward, s_next: State, done: Done) -> torch.Tensor:
        target = r + done.logical_not() * self.gamma * self.nn(s_next).max()
        return target

    def reset(self) -> None:
        self.steps = 0

    def is_full(self) -> bool:
        return len(self.buffer) >= self.bs

class OffPolicyTrainer(Trainer):

    def __init__(
        self, 
        swap_interval: int, 
        replay_buffer: ReplayBuffer,
        network: Network,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        gamma: float,
        steps_per_update: int
    ):
        super().__init__(replay_buffer, network, optimizer, batch_size, gamma, steps_per_update)

        self.swap_interval = swap_interval
        self.target_network : Network = network.copy()
    
    def step(self) -> torch.Tensor:
        if self.steps % self.swap_interval == 0:
            self.target_network = self.nn.copy()
        loss = super().step()
        return loss
    
    def target(self, r:Reward, s_next: State, done: Done) -> torch.Tensor:
        q : torch.Tensor = self.target_network(s_next)
        q_max = torch.max(q, dim=1).values
        target = r + done.logical_not() * self.gamma * q_max
        return target
