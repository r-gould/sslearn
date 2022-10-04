import torch
import torch.nn as nn

from typing import Optional

from .finetune_model import _FinetuneModel

class Classifier(_FinetuneModel):

    name = "classifier"

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        head: Optional[nn.Module] = None,
        freeze: bool = True,
    ):
        super().__init__(encoder, freeze)
        
        if head is None:
            head = nn.Sequential(
                nn.Linear(encoder.encode_dim, encoder.encode_dim),
                nn.ReLU(),
                nn.Linear(encoder.encode_dim, num_classes),
            )

        self.head = head
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, images):

        encodings = self.encoder(images)
        return self.head(encodings)

    def encode(self, images):

        return self.encoder(images)

    def step(self, images, labels):

        logits = self.forward(images)
        return self.loss_func(logits, labels)