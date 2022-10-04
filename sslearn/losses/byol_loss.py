import torch
import torch.nn as nn
import torch.nn.functional as F

class BYOLLoss:

    def loss(self, pred, target):
        """
        pred: (B, head_dim)
        target: (B, head_dim)
        """

        pred_unit = F.normalize(pred, dim=-1).unsqueeze(1)
        target_unit = F.normalize(target, dim=-1).unsqueeze(-1)

        losses = -2 * torch.bmm(pred_unit, target_unit)
        return losses.mean()


    def __call__(self, *args, **kwargs):

        return self.loss(*args, **kwargs)