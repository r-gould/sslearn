import torch
import torch.nn as nn

from .layers import MultiHeadAttention, AddNorm, FeedForward

class Encoder(nn.Module):

    def __init__(self, num_heads, encode_dim, val_dim, key_dim, mlp_dim, dropout):

        super().__init__()

        self.attention = MultiHeadAttention(num_heads, encode_dim, val_dim, key_dim)
        self.add_norm_a = AddNorm(encode_dim, dropout)

        self.feed_forward = FeedForward(encode_dim, mlp_dim)
        self.add_norm_b = AddNorm(encode_dim, dropout)

    def forward(self, encoder_in):
        
        attn = self.attention(encoder_in, encoder_in, encoder_in)
        attn = self.add_norm_a(attn, encoder_in)

        out = self.feed_forward(attn)
        return self.add_norm_b(out, attn)