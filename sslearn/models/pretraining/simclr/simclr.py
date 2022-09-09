import torch
import torch.nn as nn

from typing import Optional, Callable
from torchvision import transforms

from ..pretrain_model import PretrainModel
from ....losses.nt_xent import NTXent
from .color_distortion import color_distortion

class SimCLR(PretrainModel):

    name = "simclr"

    # maybe try RandomResizedCrop(32, ...) instead?

    default_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        color_distortion(),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    def __init__(
        self,
        encoder: nn.Module,
        head_dim: int,
        temperature: float,
        augment: Optional[Callable] = None
    ):
        super().__init__()

        if augment is None:
            augment = self.default_augment

        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.encode_dim, encoder.encode_dim),
            nn.ReLU(),
            nn.Linear(encoder.encode_dim, head_dim),
        )
        self.augment = augment
        self.loss_func = NTXent(temperature)

    def forward(self, x):

        encodings = self.encoder(x)
        return self.head(encodings)

    def step(self, x: torch.Tensor):

        x_aug1, x_aug2 = self.augment(x), self.augment(x)
        #encs1, encs2 = self.encoder(x_aug1), self.encoder(x_aug2)
        z1, z2 = self.forward(x_aug1), self.forward(x_aug2)
        return self.loss_func(z1, z2)