import torch

from ..models import Model

class Validator:

    metric_str = "metric"

    @torch.no_grad()
    def validate(self, model: Model):

        raise NotImplementedError()

    def __call__(self, *args, **kwargs):

        return self.validate(*args, **kwargs)