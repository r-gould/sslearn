import torch

from typing import Dict
from torch.utils.data import DataLoader

from ..models import Model

class Validator:

    metric_str = "metric"

    """def __init__(self, dataloaders: Dict[str, DataLoader], device: str = "cuda"):

        raise NotImplementedError()"""

    @torch.no_grad()
    def validate(self, model: Model):

        raise NotImplementedError()

    def __call__(self, *args, **kwargs):

        return self.validate(*args, **kwargs)