import torch
import numpy as np
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, encode_dim, val_dim, key_dim, masked=False):
        
        super().__init__()

        self.num_heads = num_heads

        self.linear_V = nn.Linear(encode_dim, num_heads * key_dim)
        self.linear_K = nn.Linear(encode_dim, num_heads * key_dim)
        self.linear_Q = nn.Linear(encode_dim, num_heads * key_dim)

        self.attention = ScaledDotProductAttention(key_dim, masked)

        self.output = nn.Linear(num_heads * val_dim, encode_dim)

    def forward(self, Q, K, V):
        
        inp = (self.linear_Q(Q), self.linear_K(K), self.linear_V(V))
        inp = map(self.split, inp)

        attn = self.attention(*inp)
        out = self.concat(attn)
        return self.output(out)

    def split(self, batch):
        # batch of shape (batch_size, seq_len, num_heads*d)
        
        batch_size, seq_len, _ = batch.shape
        batch = batch.reshape(batch_size, seq_len, self.num_heads, -1)
        return batch.permute(0, 2, 1, 3)

    def concat(self, batch):
        # batch of shape (batch_size, num_heads, seq_len, d_v)

        batch_size, _, seq_len, _ = batch.shape
        batch = batch.permute(0, 2, 1, 3)
        return batch.reshape(batch_size, seq_len, -1)

class ScaledDotProductAttention(nn.Module):

    def __init__(self, key_dim, masked=False):
        
        super().__init__()

        self.key_dim = key_dim
        self.masked = masked

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):

        scores = Q @ K.transpose(-1, -2) / np.sqrt(self.key_dim)

        if self.masked:
            scores = self.apply_attn_mask(scores)

        return self.softmax(scores) @ V

    def apply_attn_mask(self, scores):

        _, _, seq_len_out, seq_len_in = scores.shape

        mask = torch.triu(torch.ones(seq_len_out, seq_len_in), diagonal=1).unsqueeze(0).unsqueeze(0)
        mask = mask.to(mask.device)
            
        return scores - 1e9 * mask