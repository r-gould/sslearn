import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader

from ...models import _Model
from ...utils import cosine_similarity
from .validator import _Validator


class TopKNN(_Validator):

    metric_str = "KNN accuracy"

    def __init__(self, dataloaders: Dict[str, DataLoader], device: str = "cuda"):

        self.index_loader, self.valid_loader = dataloaders["index"], dataloaders["valid"]
        self.device = device

    @torch.no_grad()
    def validate(self, model: _Model):

        index_list, label_list = [], []

        for images, labels in self.index_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)
            encodings = model.encoder(images)

            index_list.append(encodings)
            label_list.append(labels)

        index = torch.cat(index_list, dim=0) # (bank_size, encode_dim)
        index_labels = torch.cat(label_list, dim=0) # (bank_size,)

        correct, total = 0, 0

        for images, labels in self.valid_loader:
            
            images = images.to(self.device)
            labels = labels.to(self.device)

            encodings = model.encoder(images) # (batch_size, encode_dim)

            sims = cosine_similarity(encodings, index) # (batch_size, bank_size)
            top_idxs = torch.argmax(sims, dim=-1) # (batch_size,)
            pred_labels = index_labels[top_idxs] # (batch_size,)

            correct += torch.sum(pred_labels == labels).item()
            total += len(images)

        return correct / total