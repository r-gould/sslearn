import torch
import torch.nn as nn

from .channel_pad import ChannelPad
from .norm import Norm

class Residual(nn.Module):

    SHORTCUTS = ("projection", "padding")

    def __init__(
        self,
        c_in: int,
        c_out: int,
        shortcut_type: str = "projection",
        norm_type: str = "layer",
    ):
        super().__init__()

        assert shortcut_type in self.SHORTCUTS, f"Provided shortcut_type '{shortcut_type}' is not one of {self.SHORTCUTS}."

        input_stride = (1, 1) if c_in == c_out else (2, 2)

        self.pre_res = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=input_stride, padding=(1, 1)),
            Norm(c_out, norm_type),
            nn.ReLU(),

            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=(1, 1)),
            Norm(c_out, norm_type),
            nn.ReLU(),
        )

        if c_in != c_out:
            if shortcut_type == "padding":
                self.shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=(1, 1), stride=(2, 2)),
                    ChannelPad(c_out),
                )
            elif shortcut_type == "projection":
                self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(2, 2))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.pre_res(x)
        return out + self.shortcut(x)

"""bs = 8
c = 128
h = 33
w = h
a = torch.randn(bs, c, h, w)
res = Residual(c, 256, shortcut_type="padding")
print(res(a).shape)"""