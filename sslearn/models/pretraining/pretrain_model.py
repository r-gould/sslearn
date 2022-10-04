import torch

from torchvision import transforms

from ..model import _Model

class _PretrainModel(_Model):
    """
    Abstract base class for pretraining models.
    """

    name = "pretrain"

    default_augment = None

    def step(self, x):

        raise NotImplementedError()

    @classmethod
    def _init_augment(cls, augment):

        if augment is None:
            augment = cls.default_augment

        return cls._batchwise_augment(augment)
    
    @staticmethod
    def _batchwise_augment(augment):

        return transforms.Lambda(
            lambda batch: torch.stack([augment(x) for x in batch])
        )