import torch
import torch.nn as nn

from .layers import MultiHeadAttention, FeedForward

class Encoder(nn.Module):

    def __init__(self, num_heads, encode_dim, val_dim, key_dim, mlp_dim, dropout):

        super().__init__()

        self.norm_a = nn.LayerNorm(encode_dim)
        self.attention = MultiHeadAttention(num_heads, encode_dim, val_dim, key_dim, dropout)

        self.norm_b = nn.LayerNorm(encode_dim)
        self.feed_forward = FeedForward(encode_dim, mlp_dim, dropout)

    def forward(self, encoder_in):
        
        normed_in = self.norm_a(encoder_in)
        attn = self.attention(normed_in, normed_in, normed_in)
        attn += encoder_in

        normed_attn = self.norm_b(attn)
        out = self.feed_forward(normed_attn)
        return out + attn