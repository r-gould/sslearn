import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from copy import deepcopy
from typing import Optional, Callable

from ..pretrain_model import PretrainModel
from .queue import Queue
from .info_nce import InfoNCE

class MoCo(PretrainModel):

    name = "moco"

    def __init__(self, encoder: nn.Module, queue_size: int, momentum: float, temperature: float, augment: Optional[Callable] = None):

        super().__init__()

        if augment is None:
            augment = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        self.encoder = encoder
        self.momentum_encoder = deepcopy(encoder)
        self.augment = augment
        self.encode_dim = encoder.encode_dim
        self.momentum = momentum

        self.queue = Queue(queue_size, self.encode_dim)
        self.loss_func = InfoNCE(temperature)

    """def forward(self, x):

        return self.encoder(x)"""

    def encode(self, x: torch.Tensor):

        return self.encoder(x)

    def step(self, x: torch.Tensor):
        # x of shape (batch_size, *)
        x_query = self.augment(x) # of shape (batch_size, *)
        x_key_pos = self.augment(x) # of shape (batch_size, *)

        query = self.encoder(x_query) # of shape (batch_size, encode_dim)
        with torch.no_grad():
            key_pos = self.momentum_encoder(x_key_pos) # of shape (batch_size, encode_dim)

        query = F.normalize(query, dim=-1)
        key_pos = F.normalize(key_pos, dim=-1)
        
        keys_neg = self.queue.queue.to(x.device) # of shape (K, encode_dim)
        self._save_for_update(key_pos)

        return self.loss_func(query, key_pos, keys_neg)

    def update(self):

        self._momentum_update()

        key_pos, = self._load_for_update()
        self.queue.enqueue_dequeue(key_pos)

    def _momentum_update(self):

        for (enc_param, mo_param) in zip(self.encoder.parameters(), 
                                         self.momentum_encoder.parameters()):

            mo_param = self.momentum * mo_param + (1 - self.momentum) * enc_param