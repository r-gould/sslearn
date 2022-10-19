import math
import torch
import torch.nn as nn

from copy import deepcopy
from typing import Optional, Callable
from torchvision import transforms

from .head import BYOLHead
from .. import _PretrainModel
from ....losses import BYOLLoss
from .. import SimCLR
from ....archs import _Arch

class BYOL(_PretrainModel):

    name = "byol"

    DEFAULT_AUGMENT = SimCLR.DEFAULT_AUGMENT

    def __init__(
        self,
        encoder: _Arch,
        total_iters: int,
        hidden_dim: int,
        head_dim: int,
        decay_base: float,
        augment: Optional[Callable] = None,
    ):
        super().__init__(encoder)

        self.online_network = nn.Sequential(
            encoder,
            BYOLHead(encoder.encode_dim, hidden_dim, head_dim),
        )
        self.predictor = BYOLHead(head_dim, hidden_dim, head_dim)

        self.target_network = deepcopy(self.online_network)
        self._freeze_target_network()

        self.decay_base = decay_base
        self.decay_rate = decay_base

        self.augment = self._init_augment(augment)
        self.loss_func = BYOLLoss()

        self.total_iters = total_iters
        self.iters_done = 0

    def step(self, x: torch.Tensor) -> torch.Tensor:

        x_aug_a, x_aug_b = self.augment(x), self.augment(x)
        
        pred_a = self.predict(x_aug_a)
        with torch.no_grad():
            target_b = self.target_network(x_aug_b)
        loss_a = self.loss_func(pred_a, target_b)

        pred_b = self.predict(x_aug_b)
        with torch.no_grad():
            target_a = self.target_network(x_aug_a)
        loss_b = self.loss_func(pred_b, target_a)

        return loss_a + loss_b

    def predict(self, x_aug):
        
        projection = self.online_network(x_aug)
        return self.predictor(projection)
    
    def update(self):

        # running average update target
        self.iters_done += 1
        self._update_target_network()
        self._update_decay_rate()

    def _update_target_network(self):

        for online_param, target_param in zip(self.online_network.parameters(),
                                              self.target_network.parameters()):

            target_param = self.decay_rate * target_param + (1 - self.decay_rate) * online_param

    def _update_decay_rate(self):
        
        coeff = (math.cos(math.pi * self.iters_done / self.total_iters) + 1) / 2
        self.decay_rate = 1 - (1 - self.decay_base) * coeff

    def _freeze_target_network(self):

        for param in self.target_network.parameters():
            param.requires_grad = False