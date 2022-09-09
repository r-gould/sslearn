import torch
import torch.nn as nn

class Norm(nn.Module):

    NORM_TYPES = ("layer", "batch")

    def __init__(self, channels: int, norm_type: str):

        super().__init__()

        if norm_type == "layer":
            self.norm = nn.GroupNorm(1, channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(channels)
        else:
            raise ValueError(f"Provided norm_type '{norm_type}' is not one of {self.NORM_TYPES}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.norm(x)