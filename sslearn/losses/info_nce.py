import torch
import torch.nn as nn

class InfoNCE:

    def __init__(self, temperature: float):

        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = temperature

    def loss(self, query, key_pos, keys_neg):

        # query: (batch_size, head_dim)
        # key_pos: (batch_size, head_dim)
        # keys_neg: (K, head_dim)

        # should be (batch_size, 1)
        logits_pos = torch.bmm(
            query.unsqueeze(1),
            key_pos.unsqueeze(2)
        ).squeeze(-1)

        # should be (batch_size, K)
        logits_neg = query @ keys_neg.T
        # should be (batch_size, K+1)
        logits = torch.cat([logits_pos, logits_neg],
                           dim=-1)
        logits /= self.temperature

        batch_size = len(query)
        labels = torch.zeros(batch_size).long().to(query.device)
        return self.cross_entropy(logits, labels)

    def __call__(self, *args, **kwargs):

        return self.loss(*args, **kwargs)