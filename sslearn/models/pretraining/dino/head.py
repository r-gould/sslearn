import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

class DINOHead(nn.Module):

    def __init__(
        self,
        encode_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(encode_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),

            #weight_norm(nn.Linear(bottleneck_dim, output_dim)),

            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, x):
        
        return self.network(x)