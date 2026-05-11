import torch

from cs336_basics.model import Linear


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Linear(10, 10)
        self.w2 = Linear(10, 50)
        self.w3 = Linear(50, 10)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.w1(x))
        x = self.relu(self.w2(x))
        x = self.w3(x)
        return x


class ReLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))
