import torch
import torch.nn as nn

from ..archs import _Arch

class _Model(nn.Module):
    """
    Abstract base class for models.
    """

    name = "model"

    def __init__(
        self,
        encoder: _Arch, 
        *args,
        **kwargs
    ):
        super().__init__()

        self.encoder = encoder
        self._update_cache = []

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