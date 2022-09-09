import torch
import torch.nn.functional as F

def cosine_similarity(x, y):
    
    return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T