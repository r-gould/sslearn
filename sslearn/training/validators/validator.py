import torch

from typing import Dict
from torch.utils.data import DataLoader

from ...models import _Model

class _Validator:
    """
    Abstract base class for a model validator.
    """

    metric_str = "metric"

    @torch.no_grad()
    def validate(self, model: _Model):

        raise NotImplementedError()

    def __call__(self, *args, **kwargs):

        return self.validate(*args, **kwargs)