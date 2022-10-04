import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss:

    def __init__(self, temp_s, temp_t):

        self.temp_s = temp_s
        self.temp_t = temp_t

        self.cross_entropy = nn.CrossEntropyLoss()

    def loss(self, student, teacher, center):
        """
        student: (batch_size, output_dim)
        """

        teacher = teacher.detach()
        
        student /= self.temp_s
        teacher = (teacher - center) / self.temp_t
        
        return self.cross_entropy(student, teacher)

    def __call__(self, *args, **kwargs):

        return self.loss(*args, **kwargs)