import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable
from torchvision import transforms

from .sinkhorn import sinkhorn
from .multicrop import MultiCrop
from ..simclr.color_distortion import color_distortion
from .. import _PretrainModel
from ....losses import SwAVLoss

class SwAV(_PretrainModel):

    name = "swav"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        head_dim: int,
        temperature: float,
        num_prototypes: int,
        freeze_iters: int,
        global_crop_info: list,
        local_crop_info: list,
    ):
        super().__init__()

        self.encoder = encoder
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

    def step(self, x: torch.Tensor):

        batch_size = x.shape[0]
        crops = self.multi_crop.crops(x)

        # (B*len(crops), head_dim)
        z = MultiCrop.crops_to_embeds(crops, self, self.head_dim, try_concat=False)
        z = F.normalize(z, dim=-1)

        embeds = torch.zeros(0, self.head_dim)
        codes = torch.zeros(0, self.num_prototypes, requires_grad=False)

        for i in range(self.global_count):

            global_embed = z[i*batch_size : (i+1)*batch_size]
            global_code = sinkhorn(global_embed, self.prototypes)
            for j in range(len(crops)):
                
                if i == j:
                    continue

                curr_embed = z[j*batch_size : (j+1)*batch_size]
                #loss += self.loss_func(curr_embed, global_code, self.prototypes)
                embeds = torch.cat([embeds, curr_embed], dim=0)
                codes = torch.cat([codes, global_code], dim=0)

        return self.loss_func(embeds, codes, self.prototypes)


    def update(self):

        self.iters_done += 1
        if self.iters_done == self.freeze_iters:
            print("PROTOTYPES FROZEN")
            self._freeze_prototypes()
        
        if self.iters_done <= self.freeze_iters:
            self.prototypes = F.normalize(self.prototypes, dim=0)

    def _freeze_prototypes(self):

        self.prototypes.requires_grad = False

    @staticmethod
    def _init_prototypes(head_dim, num_prototypes):
    
        return torch.randn(head_dim, num_prototypes)