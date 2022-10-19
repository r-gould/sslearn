import torch
import torch.nn as nn

from typing import Optional, Callable
from torchvision import transforms

from .. import _PretrainModel
from ....losses import NTXent
from .color_distortion import color_distortion
from ....archs import _Arch

class SimCLR(_PretrainModel):

    name = "simclr"

    DEFAULT_AUGMENT = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        color_distortion(0.5),
        transforms.RandomApply([
            transforms.GaussianBlur((32//10, 32//10), (0.1, 2)),
        ], p=0.5),
    ])

    def __init__(
        self,
        encoder: _Arch,
        hidden_dim: int,
        head_dim: int,
        temperature: float,
        augment: Optional[Callable] = None,
    ):
        super().__init__(encoder)

        self.head = nn.Sequential(
            nn.Linear(encoder.encode_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, head_dim),
        )
        
        self.augment = self._init_augment(augment)
        self.loss_func = NTXent(temperature)

    def forward(self, x):
        
        encodings = self.encoder(x)
        return self.head(encodings)

    def step(self, x: torch.Tensor) -> torch.Tensor:

        x_aug1, x_aug2 = self.augment(x), self.augment(x)
        z1, z2 = self.forward(x_aug1), self.forward(x_aug2)
        return self.loss_func(z1, z2)