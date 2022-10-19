import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from copy import deepcopy
from typing import Optional, Callable

from ....losses import InfoNCE
from .. import _PretrainModel
from .queue import Queue
from ....archs import _Arch

class MoCo(_PretrainModel):

    name = "moco"

    DEFAULT_AUGMENT = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ], p=0.8),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    VERS = ("v1.0", "v2.0")

    def __init__(
        self,
        encoder: _Arch,
        head_dim: int,
        queue_size: int,
        momentum: float,
        temperature: float,
        augment: Optional[Callable] = None,
        ver: str = "v2.0",
    ):
        super().__init__(encoder)

        if ver == "v1.0":
            self.head = nn.Linear(encoder.encode_dim, head_dim)
        elif ver == "v2.0":
            self.head = nn.Sequential(
                nn.Linear(encoder.encode_dim, encoder.encode_dim),
                nn.Linear(encoder.encode_dim, head_dim),
            )
        else:
            raise ValueError(f"Provided version '{ver}' is not one of {self.VERS}.")

        self.momentum_encoder = deepcopy(encoder)
        
        self.augment = self._init_augment(augment)

        self.momentum = momentum
        self.queue = Queue(queue_size, head_dim)
        self.loss_func = InfoNCE(temperature)

    def forward(self, x):

        encodings = self.encoder(x)
        return self.head(encodings)

    @torch.no_grad()
    def momentum_forward(self, x):

        encodings = self.momentum_encoder(x)
        return self.head(encodings)

    def step(self, x: torch.Tensor) -> torch.Tensor:

        x_query, x_key_pos = self.augment(x), self.augment(x)

        query = self.forward(x_query) # of shape (batch_size, head_dim)
        key_pos = self.momentum_forward(x_key_pos) # of shape (batch_size, head_dim)
        
        query = F.normalize(query, dim=-1)
        key_pos = F.normalize(key_pos, dim=-1)
        
        keys_neg = self.queue.queue.to(x_query.device) # of shape (K, head_dim)
        self._save_for_update(key_pos)

        return self.loss_func(query, key_pos, keys_neg)

    def update(self):

        self._momentum_update()

        key_pos, = self._load_for_update()
        self.queue.enqueue_dequeue(key_pos)

    @torch.no_grad()
    def _momentum_update(self):

        for (enc_param, mo_param) in zip(self.encoder.parameters(), 
                                         self.momentum_encoder.parameters()):

            mo_param = self.momentum * mo_param + (1 - self.momentum) * enc_param