import torch
import torch.nn as nn

from ..model import _Model
from ...archs import _Arch

class _FinetuneModel(_Model):
    """
    Abstract base class for finetuning models.
    """

    name = "finetune"

    def __init__(
        self,
        encoder: _Arch,
        freeze: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(encoder)

        if freeze:
            self._freeze()

    def forward(self, x):

        raise NotImplementedError()

    def step(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError()

    def _freeze(self):

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze(self):

        for param in self.encoder.parameters():
            param.requires_grad = True