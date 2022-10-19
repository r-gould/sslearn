import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, encode_dim, mlp_dim, dropout):

        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encode_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, encode_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, batch):

        return self.network(batch)