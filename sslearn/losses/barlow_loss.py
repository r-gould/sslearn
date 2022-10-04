import torch
import torch.nn as nn
import torch.nn.functional as F

class BarlowLoss:

    def __init__(self, lambd):

        self.lambd = lambd

    def loss(self, z_a, z_b):
        """
        z_a: (B, project_dim)
        z_b: (B, project_dim)
        """
        batch_size, project_dim = z_a.shape

        z_a_norm = F.normalize(z_a - z_a.mean(dim=0), dim=0) # maybe divide by std?
        z_b_norm = F.normalize(z_b - z_b.mean(dim=0), dim=0)

        correlations = z_a_norm.T @ z_b_norm / batch_size # (project_dim, project_dim)
        
        diag = torch.diagonal(correlations)
        
        non_diag_mask = torch.ones(project_dim, project_dim) - torch.eye(project_dim)
        non_diag = correlations * non_diag_mask.to(z_a.device)

        invariance = torch.sum((1 - diag)**2)
        redundancy = torch.sum(non_diag**2)
        
        return invariance + self.lambd * redundancy


    def __call__(self, *args, **kwargs):

        return self.loss(*args, **kwargs)