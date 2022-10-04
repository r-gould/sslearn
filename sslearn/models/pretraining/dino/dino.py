import math
import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn.utils import weight_norm
from typing import Optional, Callable
from torchvision import transforms

from .head import DINOHead
from .. import _PretrainModel
from ....losses import DINOLoss
from ..swav.multicrop import MultiCrop

class DINO(_PretrainModel):

    name = "dino"

    #default_augment = SimCLR.default_augment

    def __init__(
        self,
        encoder: nn.Module,
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
        super().__init__()

        self.encoder = encoder

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



    def step(self, x: torch.Tensor):
        # x shape (batch_size, c, h, w)
        batch_size = x.shape[0]

        crops = self.multi_crop.crops(x)
        
        # (len(crops)*batch_size, output_dim)
        student_out = MultiCrop.crops_to_embeds(crops, self.student_network, self.output_dim, try_concat=False)
        
        with torch.no_grad():
            # (len(global_crops)*batch_size, output_dim)
            global_crops = crops[:self.global_count]
            teacher_out = MultiCrop.crops_to_embeds(global_crops, self.teacher_network, self.output_dim)
            self._save_for_update(teacher_out)
            #total_teacher = MultiCrop.crops_to_embeds(crops, self.teacher_network, self.output_dim, try_concat=True)
            #self._save_for_update(total_teacher)
            #teacher_out = total_teacher[:self.global_count*batch_size]

        print("TEACHER OUT:", teacher_out.requires_grad, "(SHOULD BE False)")
        
        #student = torch.zeros(0, self.output_dim)
        #teacher = torch.zeros(0, self.output_dim, requires_grad=False)
        loss = 0
        for i in range(self.global_count):

            curr_teacher = teacher_out[i*batch_size : (i+1)*batch_size]
            
            for j in range(len(crops)):

                if i == j:
                    continue

                curr_student = student_out[j*batch_size : (j+1)*batch_size]
                
                #student = torch.cat([student, curr_student], dim=0)
                #teacher = torch.cat([teacher, curr_teacher], dim=0)
                loss += self.loss_func(curr_student, curr_teacher, self.center.to(x.device))

        #return self.loss_func(student, teacher, self.center)
        return loss

    def update(self):

        teacher_out, = self._load_for_update()
        self._update_teacher()
        self._update_center(teacher_out)

    @staticmethod
    def _init_center(output_dim):

        return torch.randn(output_dim, requires_grad=False)

    def _update_teacher(self):

        for (s_param, t_param) in zip(self.student_network.parameters(),
                                      self.teacher_network.parameters()):

            t_param = self.lambd * t_param + (1 - self.lambd) * s_param

    
    """def _update_center(self, total_teacher):

        m = self.center_rate
        mean = torch.mean(total_teacher, dim=0)
        print("UPDATE CENTER MEAN SHAPE:", mean.shape)
        self.center = m * self.center + (1 - m) * mean"""

    def _update_center(self, teacher_out):

        m = self.center_rate
        mean = torch.mean(teacher_out, dim=0)
        self.center = m * self.center + (1 - m) * mean