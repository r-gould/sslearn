import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, encode_dim, mlp_dim):

        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encode_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, encode_dim),
        )

    def forward(self, batch):

        return self.network(batch)