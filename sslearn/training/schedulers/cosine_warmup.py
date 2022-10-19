import torch

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim import Optimizer

class CosineAnnealingLinearWarmup:

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps

        self._warmup_scheduler = LambdaLR(optimizer, lambda step: step / warmup_steps)
        self._cosine_scheduler = CosineAnnealingLR(optimizer, total_steps-warmup_steps, eta_min=min_lr)
        self._step_count = 0

    def step(self):

        if self._step_count < self.warmup_steps:
            self._warmup_scheduler.step()
        else:
            self._cosine_scheduler.step()

        self._step_count += 1