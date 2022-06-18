from serializable import Serializable
from abc import abstractmethod
import torch
from dataclasses import dataclass

@dataclass
class Transition:
    s_now: torch.float
    a: torch.int
    r: torch.float
    s_next: torch.float
    done: torch.bool

class ReplayBuffer(Serializable):
    
    def __init__(
        self, 
        capacity: int,
        transition: Transition
    ) -> None:
        """
        Receives an empty transition. Required to specify dimensions
        """
        super().__init__()

        self.capacity = capacity
        self.transitions = transition

    def push(self, o: Transition) -> None:
        """
        Push new transition and if the resulting buffer size is larger than the capacity, pop the first element
        """
        assert o.a.size(dim=0) == 1, "Input batch size must be 1"
        t = self.transitions # for easier 

        #i = self.pop_index() if (len(self) > self.capacity) else -1
        if (len(self) >= self.capacity):
            i = self.pop_index()
            ii = i + 1
            # I'm sorry for who ever sees this for they will see this in their 
            # nightmares. I couldn't find a better way
            t.s_now = torch.cat((t.s_now[:i], t.s_now[ii:], o.s_now))
            t.a = torch.cat((t.a[:i], t.a[ii:], o.a))
            t.r = torch.cat((t.r[:i], t.r[ii:], o.r))
            t.s_next = torch.cat((t.s_next[:i], t.s_next[ii:], o.s_next))
            t.done = torch.cat((t.done[:i], t.done[ii:], o.done))
        else:
            t.s_now = torch.cat((t.s_now, o.s_now))
            t.a = torch.cat((t.a, o.a))
            t.r = torch.cat((t.r, o.r))
            t.s_next = torch.cat((t.s_next, o.s_next))
            t.done = torch.cat((t.done, o.done))
        
    def pop_index(self) -> int:
        """Base replay buffer is a FIFO, so it pops the first element"""
        return 0

    def __len__(self) -> int:
        return self.transitions.a.size(dim=0)

    def serialize(self) -> dict:
        return {
            "capacity": self.capacity,
            "s_now_size": self.transitions.s_now.size()[1:],
            "a_size": self.transitions.a.size()[1:],
            "r_size": self.transitions.r.size()[1:],
            "s_next_size": self.transitions.s_next.size()[1:],
            "done_size": self.transitions.done.size()[1:]
        }

if __name__ == "__main__":
    transition = Transition(
        torch.rand((0, 4, 4)),
        torch.rand((0, 4)),
        torch.rand((0, 1)),
        torch.rand((0, 4, 4)),
        torch.rand((0, 1))
    )
    b = ReplayBuffer(10,transition )

    for i in range(20):

        transition = Transition(
            torch.rand((1, 4, 4)),
            torch.rand((1, 4)),
            torch.rand((1, 1)),
            torch.rand((1, 4, 4)),
            torch.rand((1, 1))
        )

        b.push(transition)
        
    print(len(b))
