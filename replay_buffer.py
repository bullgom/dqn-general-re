from serializable import Serializable
from abc import abstractmethod
import torch
from dataclasses import dataclass

@dataclass
class Transition:

    s_now : list[torch.FloatTensor]
    a: list[torch.IntTensor]
    r: list[torch.FloatTensor]
    s_next : list[torch.FloatTensor]
    done: list[torch.BoolTensor]

    def __getitem__(self, indices) -> "Transition":
        x = self.dict()
        for key, val in x.items():
            x[key] = [val[i] for i in indices]
        return Transition(**x)

    def tuple(self) -> tuple[list[torch.FloatTensor],
                             list[torch.IntTensor],
                             list[torch.FloatTensor],
                             list[torch.FloatTensor],
                             list[torch.BoolTensor]]:
        return self.s_now, self.a, self.r, self.s_next, self.done

    def push(self, transition: "Transition") -> None:
        
        for key in self.keys():
            getattr(self, key).append(getattr(transition, key))
    
    def pop(self, i: int) -> None:
        for key in self.keys():
            getattr(self, key).pop(i)
    
    def cat(self) -> "Transition":
        d = {}
        for key in self.keys():
            d[key] = torch.cat(getattr(self,key))
        return Transition(**d)
    
    def keys(self) -> list[str]:
        return ["s_now", "a", "r", "s_next", "done"]

    def dict(self) -> dict:
        return {
            "s_now": self.s_now,
            "a": self.a,
            "r": self.r,
            "s_next": self.s_next,
            "done": self.done
        }
    
    @staticmethod
    def empty() -> "Transition":
        return Transition([],[],[],[],[])

class ReplayBuffer(Serializable):
    
    def __init__(
        self, 
        capacity: int,
        device: torch.device
    ) -> None:
        """
        Receives an empty transition. Required to specify dimensions.
        Saves in CPU memory,
        Returns in device
        """
        super().__init__()

        self.device =device
        self.capacity = capacity
        self.transitions = Transition.empty()
    
    def sample(self, count: int) -> Transition:
        """
        Could return same rows multiple times. But, whatever, right? Let the py-god select it for us
        """
        indices = torch.randint(0, len(self), (count,))
        x = self.transitions[indices].cat()
        return x.tuple()

    def push(self, o: Transition) -> None:
        """
        Push new transition and if the resulting buffer size is larger than the capacity, pop the first element
        """
        assert o.a.size(dim=0) == 1, "Input batch size must be 1"

        self.transitions.push(o)

        if (len(self) >= self.capacity):
            i = self.pop_index()
            self.transitions.pop(i)
        
    def pop_index(self) -> int:
        """Base replay buffer is a FIFO, so it pops the first element"""
        return 0

    def __len__(self) -> int:
        return len(self.transitions.a)

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
    print(b.sample(1))
