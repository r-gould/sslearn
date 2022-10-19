import torch
import torch.nn as nn

class _Arch(nn.Module):
    """
    Abstract base class for encoder architecture.
    """

    def __init__(self, encode_dim, *args, **kwargs):

        super().__init__()

        self.encode_dim = encode_dim

    def forward(self, x: torch.Tensor):

        raise NotImplementedError()