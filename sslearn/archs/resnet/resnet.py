import torch
import warnings
import torch.nn as nn

from .layers import Residual, Bottleneck

class ResNet(nn.Module):
    """ResNet architecture.

    Args:
        block_counts (list/tuple): list of form []
    """

    TYPES = ("regular", "bottleneck")

    RESNET_CONFIGS = {
        "resnet-18" : ([2, 2, 2, 2], "regular"),
        "resnet-34" : ([3, 4, 6, 3], "regular"),
        "resnet-50" : ([3, 4, 6, 3], "bottleneck"),
        "resnet-101" : ([3, 4, 23, 3], "bottleneck"),
        "resnet-152" : ([3, 8, 36, 3], "bottleneck"),
    }

    def __init__(self, channels_in, block_counts=None, res_type="regular", model_name=None,
                 shortcut_type="projection", block_scaling=4):
        
        super().__init__()
        
        if model_name:
            if block_counts:
                warnings.warn("'block_counts' and 'model_name' have both been provided, "
                              "but 'model_name' takes precedence for building the network.")

            block_counts, res_type = self.load_config(model_name)
        
        assert res_type in self.TYPES, f"Provided 'res_type' is not one of {self.TYPES}."

        self.network, self.encode_dim = self._build_network(channels_in, block_counts, res_type, shortcut_type, block_scaling)
    
    def forward(self, x):
        # INPUT BE 229x229 IN PAPER TABLE 1 EXAMPLE
        return self.network(x)

    def load_config(self, model_name):

        assert model_name in self.RESNET_CONFIGS.keys(), f"Provided 'model_name' is not one of {list(self.RESNET_CONFIGS.keys())}."
        
        return self.RESNET_CONFIGS.get(model_name)

    @classmethod
    def _build_network(cls, channels_in, block_counts, res_type, shortcut_type, block_scaling):

        assert len(block_counts) == 4

        layer_list = [
            nn.Conv2d(channels_in, 64, kernel_size=(7, 7), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        ]

        c_in = 64
        c_out = c_in

        for num_blocks in block_counts:

            for _ in range(num_blocks):

                layer = cls._build_layer(c_in, c_out, res_type, shortcut_type, block_scaling)
                layer_list.append(layer)
                c_in = c_out

            c_out *= 2

        layer_list.extend([
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            #nn.Linear(c_in, encode_dim),
        ]) 

        return nn.Sequential(*layer_list), c_in

    @staticmethod
    def _build_layer(c_in, c_out, res_type, shortcut_type, block_scaling):

        if res_type == "regular":
            return Residual(c_in, c_out, shortcut_type)

        elif res_type == "bottleneck":
            c_mid = c_out // block_scaling
            return Bottleneck(c_in, c_mid, c_out, shortcut_type)