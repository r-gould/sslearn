import math
import torch
import torch.nn as nn

from typing import Optional, Callable
from torchvision import transforms

from .. import _PretrainModel
from ....losses import BarlowLoss
from .. import SimCLR
from ....archs import _Arch

class BarlowTwins(_PretrainModel):

    name = "barlowtwins"

    #default_augment = SimCLR.default_augment
    DEFAULT_AUGMENT = SimCLR.DEFAULT_AUGMENT

    def __init__(
        self,
        encoder: _Arch,
        project_dim: int,
        lambd: float,
        augment: Optional[Callable] = None,
    ):
        super().__init__(encoder)

        self.projector = nn.Sequential(
            nn.Linear(encoder.encode_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),

            nn.Linear(project_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),

            nn.Linear(project_dim, project_dim),
        )

        self.augment = self._init_augment(augment)
        self.loss_func = BarlowLoss(lambd)


    def forward(self, x: torch.Tensor):

        encodings = self.encoder(x)
        return self.projector(encodings)
        
    def step(self, x: torch.Tensor) -> torch.Tensor:

        x_aug_a, x_aug_b = self.augment(x), self.augment(x)
        z_a, z_b = self.forward(x_aug_a), self.forward(x_aug_b)
        return self.loss_func(z_a, z_b)