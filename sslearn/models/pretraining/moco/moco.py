import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from copy import deepcopy
from typing import Optional, Callable

from ....losses import InfoNCE
from .. import _PretrainModel
from .queue import Queue

class MoCo(_PretrainModel):

    name = "moco"

    default_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
    ])

    VERS = ("v1.0", "v2.0")

    def __init__(
        self,
        encoder: nn.Module,
        head_dim: int,
        queue_size: int,
        momentum: float,
        temperature: float,
        augment: Optional[Callable] = None,
        ver: str = "v2.0",
    ):
        super().__init__()

        if ver == "v1.0":
            self.head = nn.Linear(encoder.encode_dim, head_dim)
        elif ver == "v2.0":
            self.head = nn.Sequential(
                nn.Linear(encoder.encode_dim, encoder.encode_dim),
                nn.Linear(encoder.encode_dim, head_dim),
            )
        else:
            raise ValueError(f"Provided version '{ver}' is not one of {self.VERS}.")

        self.encoder = encoder
        self.momentum_encoder = deepcopy(encoder)
        
        self.augment = self._init_augment(augment)

        self.momentum = momentum
        self.queue = Queue(queue_size, head_dim)
        self.loss_func = InfoNCE(temperature)

    def forward(self, x):

        encodings = self.encoder(x)
        return self.head(encodings)

    def momentum_forward(self, x):

        encodings = self.momentum_encoder(x)
        return self.head(encodings)

    def encode(self, x: torch.Tensor):

        return self.encoder(x)

    def step(self, x: torch.Tensor):
        # x of shape (batch_size, *)
        x_query = self.augment(x) # of shape (batch_size, *)
        x_key_pos = self.augment(x) # of shape (batch_size, *)

        query = self.forward(x_query) # of shape (batch_size, head_dim)
        with torch.no_grad():
            key_pos = self.momentum_forward(x_key_pos) # of shape (batch_size, head_dim)

        query = F.normalize(query, dim=-1)
        key_pos = F.normalize(key_pos.detach(), dim=-1)
        
        keys_neg = self.queue.queue.to(x.device) # of shape (K, head_dim)
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