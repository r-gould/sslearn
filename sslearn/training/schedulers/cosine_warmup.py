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
        #self.lrs = [0]

    def step(self):

        if self._step_count < self.warmup_steps:
            self._warmup_scheduler.step()
        else:
            self._cosine_scheduler.step()

        self._step_count += 1

        #for param_group in self.optimizer.param_groups:
        #self.lrs.append(self.optimizer.param_groups[0]['lr'])



"""import torch.nn as nn
net = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Softmax(),
)
optim = torch.optim.Adam(net.parameters(), lr=1)
scheduler = CosineAnnealingLinearWarmup(optim, warmup_steps=100, total_steps=1000)

for _ in range(1000):
    optim.zero_grad()
    optim.step()

    scheduler.step()

import matplotlib.pyplot as plt

plt.plot(scheduler.lrs)
plt.show()"""