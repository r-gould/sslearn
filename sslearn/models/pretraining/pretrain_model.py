import torch

from torchvision import transforms

from ..model import _Model

class _PretrainModel(_Model):
    """
    Abstract base class for pretraining models.
    """

    name = "pretrain"

    DEFAULT_AUGMENT = transforms.Compose([])

    def step(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError()

    @classmethod
    def _init_augment(cls, augment, batchwise: bool = True):

        if augment is None:
            augment = cls.DEFAULT_AUGMENT

        if batchwise:
            return transforms.Lambda(
                lambda batch: torch.stack([augment(x) for x in batch])
            )
        
        return augment