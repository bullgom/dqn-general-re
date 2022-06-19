import torch
import torchvision.transforms as tf
from abc import ABC, abstractmethod
from typing import Any

class Preprocessing:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def size(self, size: torch.Size) -> torch.Size:
        """Default returns same"""
        return size
    
    def ep_reset(self) -> None:
        """Episode reset"""
        pass

class AddBatchDim(Preprocessing):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)
    
    def size(self, size: torch.Size) -> torch.Size:
        return torch.Size([1]) + size
    
class ToDevice(Preprocessing):

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device)

class WrappedProcessing(Preprocessing):

    def __init__(self, transform: Any) -> None:
        super().__init__()
        self.inner = transform
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

class Resize(WrappedProcessing):

    def __init__(self, img_size: tuple) -> None:
        self.inner = tf.Resize(img_size)
        self.img_size = torch.Size(img_size)
    
    def size(self, size: torch.Size) -> torch.Size:
        return size[:-len(self.img_size)] + self.img_size

class ToTensor(WrappedProcessing):

    def __init__(self) -> None:
        self.inner = tf.ToTensor()
    
    def size(self, size: torch.Size) -> torch.Size:
        assert len(size) == 3, f"This method assumes input to be 3-dim, got {len(size)}"
        w, h, c = size
        return torch.Size([c, w, h])

class MultiFrame(Preprocessing):

    def __init__(self, frames:int) -> None:
        super().__init__()

        self.frames = frames
        self.memory = []

    def ep_reset(self) -> None:
        self.memory.clear()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 4, f"Requires 4D tensor"
        if not self.memory:
            for i in range(self.frames):
                self.memory.append(torch.zeros(x.size()))
        
        self.memory.append(x)
        self.memory.pop(0)

        return torch.cat(self.memory, dim=1)

    def size(self, size: torch.Size) -> torch.Size:
        return torch.Size([size[0], size[1] * self.frames, *size[2:]])
