import torch

from typing import Type
from torchvision.datasets import VisionDataset

from ..models import Model

class Validator:

    metric_str = "metric"

    def __init__(self, dataset_cls: Type[VisionDataset], device: str = "cuda"):

        raise NotImplementedError()

    @torch.no_grad()
    def validate(self, model: Model):

        raise NotImplementedError()

    def __call__(self, *args, **kwargs):

        return self.validate(*args, **kwargs)