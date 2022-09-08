import torch
import torch.nn as nn

class InfoNCE:

    def __init__(self, temperature: float):

        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = temperature

    def loss(self, query, key_pos, keys_neg):

        # query: (batch_size, encode_dim)
        # key_pos: (batch_size, encode_dim)
        # keys_neg: (K, encode_dim)

        # below should be (batch_size, 1)
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
        #print(logits.dtype, labels.dtype)
        return self.cross_entropy(logits, labels)

    """def loss(self, query, keys, pos_idx=0):

        # query of shape (batch_size, encode_dim)
        # keys of shape (batch_size, K+1, encode_dim)
        query = query.unsqueeze(1)
        keys = keys.transpose(1, 2)
        dot_prods = torch.bmm(query, keys).squeeze() # of shape (batch_size, K+1)
        
        # of shape (batch_size,)
        numerator = torch.exp(dot_prods[:, pos_idx] / self.temperature)
         # of shape (batch_size,)
        denominator = torch.sum(torch.exp(dot_prods / self.temperature), 
                                dim=-1)

        return -torch.log(numerator / denominator)"""

    def __call__(self, query, key_pos, keys_neg):

        return self.loss(query, key_pos, keys_neg)