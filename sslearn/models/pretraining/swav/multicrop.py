import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..simclr.color_distortion import color_distortion

class MultiCrop:

    GLOBAL_SCALE = (0.4, 1)
    LOCAL_SCALE = (0.1, 0.4)

    def __init__(self, global_crop_info, local_crop_info, batchwise: bool = True):

        self.global_augments = self.create_augments(global_crop_info, self.GLOBAL_SCALE, batchwise)
        self.local_augments = self.create_augments(local_crop_info, self.LOCAL_SCALE, batchwise)
        
        self.augments = self.global_augments + self.local_augments
        self.crop_info = global_crop_info + local_crop_info

    def __call__(self, image):
        
        return self.crops(image)

    def crops(self, image):

        crops = []

        for i, (num_crops, _) in enumerate(self.crop_info):

            augment = self.augments[i]

            for _ in range(num_crops):
                crops.append(augment(image))

        return crops

    @staticmethod
    def create_augments(crop_info, scale, batchwise):
        
        augments = []

        for (_, size) in crop_info:
            augment = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                color_distortion(s=0.5),
            ])

            if batchwise:
                augment_for_batch = transforms.Lambda(
                    lambda batch: torch.stack([augment(x) for x in batch])
                )
                augments.append(augment_for_batch)
                continue
                
            augments.append(augment)

        return augments
        

    """@staticmethod
    def crops_to_embeds(crops: list, network: nn.Module, output_dim: int):
        
        # Takes crops to their embeddings
        # shape (len(crops)*batch_size, head_dim)
        
        
        out = torch.zeros(0, output_dim)

        prev_shape = crops[0].shape[1:]
        batch_concat = torch.zeros(0, *prev_shape)

        for crop in crops:

            if crop.shape[1:] == prev_shape:
                batch_concat = torch.cat([batch_concat, crop], dim=0)
                continue
            
            embeds = network(batch_concat)
            out = torch.cat([out, embeds], dim=0)
            
            prev_shape = crop.shape[1:]
            batch_concat = crop

        embeds = network(batch_concat)
        return torch.cat([out, embeds], dim=0)"""