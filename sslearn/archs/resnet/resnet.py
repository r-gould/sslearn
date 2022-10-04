import torch
import warnings
import torch.nn as nn

from typing import Optional, Tuple

from .layers import Residual, Bottleneck

class ResNet(nn.Module):
    """
    ResNet architecture.
    """

    TYPES = ("regular", "bottleneck")

    RESNET_CONFIGS = {
        "resnet-18" : ((2, 2, 2, 2), "regular"),
        "resnet-34" : ((3, 4, 6, 3), "regular"),
        "resnet-50" : ((3, 4, 6, 3), "bottleneck"),
        "resnet-101" : ((3, 4, 23, 3), "bottleneck"),
        "resnet-152" : ((3, 8, 36, 3), "bottleneck"),
    }

    def __init__(
        self,
        channels_in: int,
        block_counts: Optional[Tuple[int]] = None,
        res_type: str = "regular",
        model_name: Optional[str] = None,
        shortcut_type: str = "projection",
        block_scaling: Optional[int] = None,
        out_scaling: Optional[int] = None,
        cifar10: bool = False,
    ):
        super().__init__()
        
        if model_name:
            if block_counts:
                warnings.warn("'block_counts' and 'model_name' have both been provided, "
                              "but 'model_name' takes precedence for building the network.")

            block_counts, res_type = self.load_config(model_name)
        
        assert res_type in self.TYPES, f"Provided res_type '{res_type}' is not one of {self.TYPES}."

        if block_scaling is None:
            block_scaling = 2

        if out_scaling is None:
            out_scaling = 1 if res_type == "regular" else 4

        self.network, self.encode_dim = self._build_network(channels_in, block_counts, res_type, shortcut_type, block_scaling, out_scaling, cifar10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.network(x)

    def load_config(self, model_name: str) -> Tuple[Tuple[int], str]:

        assert model_name in self.RESNET_CONFIGS.keys(), f"Provided model_name '{model_name}' is not one of {list(self.RESNET_CONFIGS.keys())}."
        
        return self.RESNET_CONFIGS.get(model_name)

    @classmethod
    def _build_network(
        cls,
        channels_in: int,
        block_counts: Tuple[int],
        res_type: str,
        shortcut_type: str,
        block_scaling: int,
        out_scaling: int,
        cifar10: bool,
    ):
        assert len(block_counts) == 4

        if cifar10:
            layer_list = [
                nn.Conv2d(channels_in, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ]

        else:
            layer_list = [
                nn.Conv2d(channels_in, 64, kernel_size=(7, 7), stride=(2, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
            ]

        c_in = 64
        c_out = c_in * out_scaling

        for num_blocks in block_counts:

            for _ in range(num_blocks):

                layer = cls._build_layer(c_in, c_out, res_type, shortcut_type, block_scaling, out_scaling)
                layer_list.append(layer)
                c_in = c_out

            c_out *= block_scaling

        layer_list.extend([
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        ])

        return nn.Sequential(*layer_list), c_in

    @staticmethod
    def _build_layer(
        c_in: int,
        c_out: int,
        res_type: str,
        shortcut_type: str,
        block_scaling: int,
        out_scaling: int,
    ):

        if res_type == "regular":
            return Residual(c_in, c_out, shortcut_type)

        elif res_type == "bottleneck":
            c_mid = c_out // out_scaling
            return Bottleneck(c_in, c_mid, c_out, shortcut_type)