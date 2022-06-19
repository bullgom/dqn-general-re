import torch
import torch.nn as nn
import torch.nn.functional as F
from util import conv2d_output_size

class Network(torch.nn.Module):

    def __init__(
        self,
        input_size: tuple[int,int,int],
        output_size: int
    ):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        b, c, w, h = input_size

        c1 = 16
        c2 = 16
        c3 = 16
        self.conv1 = nn.Conv2d(c, c1, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(c3)

        size = torch.tensor([w, h])
        size = conv2d_output_size(size, 5, 2)
        size = conv2d_output_size(size, 5, 2)
        size = conv2d_output_size(size, 5, 2)
        w, h = size
        lin_size = w * h * c3

        self.head = nn.Linear(lin_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)
    
    def copy(self) -> "Network":
        x = Network(self.input_size, self.output_size)
        x.load_state_dict(self.state_dict())
        return x
        
if __name__ == "__main__":
    net = Network((2, 100, 100), 2)
    x = torch.rand((1, 2, 100, 100))
    y = net(x)
    assert y.size() == torch.Size([1, 2])
