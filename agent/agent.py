from strategy import Strategy
import torch
from plugin import Plugin
from base import Base
from my_types import State, Action

class Agent(Base):

    def __init__(
        self,
        network: torch.nn.Module,
        strategy: Strategy,
    ) -> None:
        
        self.network = network
        self.strategy = strategy
    
    def select(self, state: State) -> Action:
        with torch.no_grad():
            q = self.network(state)
            return self.strategy.select(q).unsqueeze(0)

    def step(self) -> None:
        self.strategy.step()
    
    def reset(self) -> None:
        self.strategy.reset()

