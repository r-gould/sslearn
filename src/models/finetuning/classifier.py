import torch
import torch.nn as nn

from .finetune_model import FinetuneModel

class Classifier(FinetuneModel):

    def __init__(self, encoder: nn.Module, num_classes: int):

        super().__init__(encoder)

        self.head = nn.Linear(encoder.encode_dim, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, images):

        encodings = self.encoder(images)
        return self.head(encodings)

    def encode(self, images):

        return self.encoder(images)

    def step(self, images, labels):

        logits = self.forward(images)
        return self.loss_func(logits, labels)