import torch
import torch.nn as nn

from typing import Optional

from .encoder import Encoder

class ViT(nn.Module):

    VIT_CONFIGS = {
        "vit-s" : (12, 6, 384, 1536),
        "vit-b" : (12, 12, 768, 3072),
        "vit-l" : (24, 16, 1024, 4096),
        "vit-h" : (32, 16, 1280, 5120),
    }

    def __init__(
        self,
        image_shape: tuple,
        patch_shape: tuple,
        model_name: Optional[str] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        encode_dim: Optional[int] = None,
        mlp_dim: Optional[int] = None, 
        dropout: float = 0.0,
        channels: int = 3,
    ):
        super().__init__()
        
        if model_name:
            num_layers, num_heads, encode_dim, mlp_dim = self.load_config(model_name)

        if None in [num_layers, num_heads, encode_dim, mlp_dim]:
            raise ValueError("Either a valid 'model_name' or all arguments must be provided.")

        self.encode_dim = encode_dim
        self.embedding = self._init_embedding(encode_dim, channels, patch_shape)
        
        # (num_patches + 1, encode_dim)
        self.pos_embeddings = nn.Parameter(
            self._init_pos_embeddings(encode_dim, image_shape, patch_shape)
        )
        
        # (1, encode_dim, 1)
        self.class_token = nn.Parameter(
            self._init_class_token(encode_dim).unsqueeze(0).unsqueeze(0)
        )

        self.drop = nn.Dropout(dropout)
        
        val_dim = key_dim = encode_dim // num_heads
        self.encoder_stack = nn.Sequential(
            *[Encoder(num_heads, encode_dim, val_dim, key_dim, mlp_dim, dropout) 
              for _ in range(num_layers)]
        )

        self.image_shape = image_shape
        self.patch_shape = patch_shape

    def forward(self, images):
        # images of shape (batch_size, channels, height, width)

        assert images.shape[-2:] == self.image_shape

        batch_size = images.shape[0]

        # (batch_size, num_patches, channels*patch_height*patch_width)
        flat_patches = self.split_images(images, self.patch_shape, flatten=True)

        # (batch_size, num_patches, encode_dim)
        embeds = self.embedding(flat_patches)

        # (batch_size, 1, encode_dim)
        class_token = self.class_token.expand(batch_size, -1, -1)
        # (batch_size, num_patches + 1, encode_dim)
        embeds = torch.concat([class_token, embeds], dim=1)
        embeds += self.pos_embeddings
        embeds = self.drop(embeds) 

        # (batch_size, num_patches + 1, encode_dim)
        encodings = self.encoder_stack(embeds)

        # returns (batch_size, encode_dim)
        return encodings[:, 0, :]



    def load_config(self, model_name: str):

        assert model_name in self.VIT_CONFIGS.keys(), f"Provided model_name '{model_name}' is not one of {list(self.VIT_CONFIGS.keys())}."

        return self.VIT_CONFIGS.get(model_name)

    @staticmethod
    def split_images(images, patch_shape, flatten: bool):
        # images of shape (batch_size, channels, height, width)

        # shape (batch_size, num_patches, channels, patch_height, patch_width) if not flattened,
        # else (batch_size, num_patches, channels*patch_height*patch_width)

        patch_height, patch_width = patch_shape

        batch_size, channels, height, width = images.shape

        #assert (height % patch_height == 0) and (width % patch_width == 0)

        total_rows = height // patch_height
        total_cols = width // patch_width

        out = torch.Tensor(batch_size, 0, channels, patch_height, patch_width).to(images.device)
        for row in range(total_rows):
            for col in range(total_cols):
                
                row_l = row * patch_height
                row_u = row_l + patch_height

                col_l = col * patch_width
                col_u = col_l + patch_width

                img = images[:, :, row_l:row_u, col_l:col_u].unsqueeze(1)
                out = torch.concat([out, img], dim=1)

        if flatten:
            out = torch.flatten(out, start_dim=2)
        
        return out

    @staticmethod
    def _init_embedding(encode_dim, channels, patch_shape):

        patch_height, patch_width = patch_shape
        flat_dim = channels * patch_height * patch_width
        return nn.Linear(flat_dim, encode_dim)

    @staticmethod
    def _init_pos_embeddings(encode_dim, image_shape, patch_shape):

        height, width = image_shape
        patch_height, patch_width = patch_shape

        assert (height % patch_height == 0) and (width % patch_width == 0)

        num_patches = (height // patch_height) * (width // patch_width)

        # + 1 is for class token
        return torch.randn(num_patches + 1, encode_dim)

    @staticmethod
    def _init_class_token(encode_dim):

        return torch.randn(encode_dim)