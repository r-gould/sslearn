import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXent:

    def __init__(self, temperature: float):

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def loss(self, batch1, batch2, large=1e6):
        
        assert batch1.shape == batch2.shape

        bs, _ = batch1.shape
        batch = torch.cat([batch1, batch2], dim=0) # (2bs, head_dim)
        batch = F.normalize(batch, dim=-1)
        sims = batch @ batch.T # (2bs, 2bs)
        mask = torch.eye(2*bs).to(batch.device)
        sims = sims - large * mask
        labels = torch.cat([torch.arange(bs, 2*bs), torch.arange(bs)], dim=0).to(batch.device)

        return self.cross_entropy(sims / self.temperature, labels)

    def __call__(self, *args, **kwargs):

        return self.loss(*args, **kwargs)