import torch
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn import sinkhorn
from .multicrop import MultiCrop
from .. import _PretrainModel
from ....losses import SwAVLoss
from ....archs import _Arch

class SwAV(_PretrainModel):

    name = "swav"

    def __init__(
        self,
        encoder: _Arch,
        hidden_dim: int,
        head_dim: int,
        temperature: float,
        num_prototypes: int,
        freeze_iters: int,
        global_crop_info: list,
        local_crop_info: list,
    ):
        super().__init__(encoder)

        self.head = nn.Sequential(
            nn.Linear(encoder.encode_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, head_dim),
        )
        self.head_dim = head_dim
        self.num_prototypes = num_prototypes
        
        self.loss_func = SwAVLoss(temperature)

        self.prototypes = nn.Parameter(
            self._init_prototypes(head_dim, num_prototypes)
        )

        self.freeze_iters = freeze_iters
        self.iters_done = 0

        self.global_count = sum([count for count, _ in global_crop_info])
        self.multi_crop = MultiCrop(global_crop_info, local_crop_info)

    def forward(self, x):

        encodings = self.encoder(x)
        return self.head(encodings)
    
    def step(self, x: torch.Tensor) -> torch.Tensor:

        crops = self.multi_crop(x)

        z = []
        for crop in crops:
            embed = self.forward(crop)
            z.append(F.normalize(embed, dim=-1))

        loss = 0
        for i in range(self.global_count):

            global_embed = z[i]
            global_code = sinkhorn(global_embed, self.prototypes)
            for j in range(len(crops)):
                
                if i == j:
                    continue

                curr_embed = z[j]
                loss += self.loss_func(curr_embed, global_code, self.prototypes)

        return loss

    def update(self):

        self.iters_done += 1
        if self.iters_done == self.freeze_iters:
            self._freeze_prototypes()
        
        if self.iters_done <= self.freeze_iters:
            self.prototypes.data = F.normalize(self.prototypes.data, dim=0)

    def _freeze_prototypes(self):

        self.prototypes.requires_grad = False

    @staticmethod
    def _init_prototypes(head_dim, num_prototypes):
    
        prototypes = torch.randn(head_dim, num_prototypes)
        return F.normalize(prototypes, dim=0)