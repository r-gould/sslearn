import torch
import torch.nn as nn

from typing import Optional

from .finetune_model import _FinetuneModel
from ...archs import _Arch

class Classifier(_FinetuneModel):

    name = "classifier"

    def __init__(
        self,
        encoder: _Arch,
        hidden_dim: int,
        num_classes: int,
        head: Optional[nn.Module] = None,
        freeze: bool = True,
    ):
        super().__init__(encoder, freeze)
        
        if head is None:
            head = nn.Sequential(
                nn.Linear(encoder.encode_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

        self.head = head
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):

        encodings = self.encoder(x)
        return self.head(encodings)

    def step(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        logits = self.forward(x)
        return self.loss_func(logits, target)