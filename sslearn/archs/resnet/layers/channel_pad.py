import torch
import torch.nn as nn

class ChannelPad(nn.Module):

    def __init__(self, target_channels: int):

        super().__init__()

        self.target_channels = target_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        bs, channels, h, w = x.shape
        channel_diff = self.target_channels - channels

        assert channel_diff >= 0

        zeros = torch.zeros(bs, channel_diff, h, w)
        return torch.cat([x, zeros], dim=1)