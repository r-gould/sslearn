import torch

from ..models.finetuning import FinetuneModel
from .validator import Validator

class Accuracy(Validator):

    metric_str = "accuracy"

    def __init__(self, dataloaders, device="cuda"):

        self.valid_loader = dataloaders["valid"]
        self.device = device

    @torch.no_grad()
    def validate(self, model: FinetuneModel):

        correct, total = 0, 0

        for images, labels in self.valid_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = model.forward(images) # (batch_size, num_classes)
            pred_labels = torch.argmax(logits, dim=-1)

            correct += torch.sum(pred_labels == labels).item()
            total += len(images)

        return correct / total