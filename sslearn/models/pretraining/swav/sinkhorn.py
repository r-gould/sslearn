import torch
import torch.nn as nn

@torch.no_grad()
def sinkhorn(z: torch.Tensor, prototypes: nn.Module, eps: float = 0.05, iters: int = 3):
    """
    Produces a matrix such that for each batch, the sum over the prototype
    dimension (torch.sum(q, dim=1)) is a vector of ones (to represent 
    prototype probabilities) and the sum over the batch dimension 
    (torch.sum(q, dim=0)) gives a vector filled with the 
    value 'batch_size / num_prototypes' (which represents the equipartition 
    constraint), where the probabilities are to remain consistent with 'scores'.
    """

    q = torch.exp(z @ prototypes / eps) # shape (B, num_prototypes)
    batch_size, num_prototypes = q.shape

    q /= torch.sum(q)

    for _ in range(iters):

        q /= torch.sum(q, dim=0, keepdim=True) * num_prototypes
        q /= torch.sum(q, dim=1, keepdim=True) * batch_size
    
    return batch_size * q