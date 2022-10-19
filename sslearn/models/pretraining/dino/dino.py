import math
from sklearn.cluster import k_means
import torch
import torch.nn as nn

from copy import deepcopy
from typing import Optional, Callable
from torchvision import transforms

from .head import DINOHead
from .. import _PretrainModel
from ....losses import DINOLoss
from ..swav.multicrop import MultiCrop
from ....archs import _Arch

class DINO(_PretrainModel):

    name = "dino"

    #default_augment = SimCLR.default_augment

    def __init__(
        self,
        encoder: _Arch,
        hidden_dim: int,# = 2048,
        bottleneck_dim: int,# = 256,
        output_dim: int,# = 4096,  maybe use 65536? should give best
        temp_s: float,
        temp_t: float,
        lambd: float,
        center_rate: float,
        global_crop_info: list,
        local_crop_info: list,
    ):
        super().__init__(encoder)

        self.student_network = nn.Sequential(
            encoder,
            DINOHead(encoder.encode_dim, hidden_dim, bottleneck_dim, output_dim),
        )

        self.teacher_network = deepcopy(self.student_network)

        self.center = self._init_center(output_dim)

        self.loss_func = DINOLoss(temp_s, temp_t)

        self.global_count = sum([count for count, _ in global_crop_info])
        
        self.multi_crop = MultiCrop(global_crop_info, local_crop_info)

        self.output_dim = output_dim
        self.lambd = lambd
        self.center_rate = center_rate



    #def step(self, x: torch.Tensor):
    def step(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch_size, c, h, w)
        crops = self.multi_crop(x)
        batch_size = crops[0].shape[0]
        self.center = self.center.to(crops[0].device)

        #crops = self.multi_crop.crops(x)

        students = [None for _ in range(len(crops))]
        for i in range(len(crops)):
            crop = crops[i]
            students[i] = self.student_network(crop)

        teachers = [None for _ in range(self.global_count)]
        with torch.no_grad():
            global_crops = torch.cat(crops[:self.global_count], dim=0)
            teachers_tensor = self.teacher_network(global_crops)
            for i in range(self.global_count):
                teachers[i] = teachers_tensor[i*batch_size : (i+1)*batch_size]

        loss = 0
        for i in range(self.global_count):

            teacher = teachers[i]

            for j in range(len(crops)):

                if i == j:
                    continue

                student = students[j]

                loss += self.loss_func(student, teacher, self.center)

        self._save_for_update(teachers_tensor)
        loss /= (self.global_count * (len(crops)-1))
        return loss

    def update(self):
        teacher_out, = self._load_for_update()
        self._update_teacher()
        self._update_center(teacher_out)

    @staticmethod
    def _init_center(output_dim):
        center = torch.zeros(1, output_dim, requires_grad=False)
        center.uniform_(-1, 1)
        return center

    @torch.no_grad()
    def _update_teacher(self):

        for (s_param, t_param) in zip(self.student_network.parameters(),
                                      self.teacher_network.parameters()):

            t_param = self.lambd * t_param + (1 - self.lambd) * s_param

    @torch.no_grad()
    def _update_center(self, teacher_out):
        m = self.center_rate
        mean = torch.mean(teacher_out, dim=0, keepdim=True)
        self.center = m * self.center + (1 - m) * mean