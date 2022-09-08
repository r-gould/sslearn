import torch
import torch.nn as nn

class Model(nn.Module):

    name = "model"

    def __init__(self):

        super().__init__()
        
        self._update_cache = []

    def encode(self, x):

        raise NotImplementedError()

    def step(self, *args, **kwargs) -> torch.Tensor:

        """
        Should return a loss for a single training step.
        """

        raise NotImplementedError()

    def update(self):

        pass

    def _save_for_update(self, *args):

        self._update_cache.extend(args)

    def _load_for_update(self):

        items = self._update_cache
        self._update_cache = []
        return items