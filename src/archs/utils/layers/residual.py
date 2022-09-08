import torch
import torch.nn as nn

from .channel_pad import ChannelPad

SHORTCUTS = ["projection", "padding"]

class Residual(nn.Module):

    def __init__(self, c_in, c_out, shortcut_type="projection"):

        super().__init__()

        assert shortcut_type in SHORTCUTS, f"Provided 'shortcut_type' is not one of {SHORTCUTS}."

        input_stride = (1, 1) if c_in == c_out else (2, 2)

        self.pre_res = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=input_stride, padding=(1, 1)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),

            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c_out),
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

    def forward(self, x):

        out = self.pre_res(x)
        return out + self.shortcut(x)

"""bs = 8
c = 128
h = 33
w = h
a = torch.randn(bs, c, h, w)
res = Residual(c, 256, shortcut_type="padding")
print(res(a).shape)"""