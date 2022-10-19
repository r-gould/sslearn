import torch
import torch.nn as nn

class SwAVLoss:

    def __init__(self, temperature: float):

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    #def loss(self, z_t, z_s, q_t, q_s, prototypes):
    def loss(self, z, q, prototypes):
        """
        Z_t: (B, encode_dim)
        Z_s: (B, encode_dim)
        Q_t: (B, K)
        Q_s: (B, K)
        C: (encode_dim, K)
        """
        
        return self.cross_entropy(z @ prototypes / self.temperature, q)

    def __call__(self, *args, **kwargs):

        return self.loss(*args, **kwargs)