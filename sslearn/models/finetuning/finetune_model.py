import torch.nn as nn

from ..model import _Model

class _FinetuneModel(_Model):
    """
    Abstract base class for finetuning models.
    """

    name = "finetune"

    def __init__(self, encoder: nn.Module, freeze: bool = True, *args, **kwargs):

        super().__init__()

        self.encoder = encoder
        if freeze:
            self._freeze()

    def forward(self, x):

        raise NotImplementedError()

    def step(self, x, target):

        raise NotImplementedError()

    def _freeze(self):

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze(self):

        for param in self.encoder.parameters():
            param.requires_grad = True