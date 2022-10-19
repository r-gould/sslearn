from vit.vit import ViT

import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, vit):
        
        super().__init__()
        self.vit = vit

        self.head = nn.Sequential(
            nn.Linear(vit.encode_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):

        encodings = self.vit(x)
        return self.head(encodings)



from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

vit = ViT(image_shape=(32, 32), patch_shape=(16, 16), model_name="vit-s")

print("MY VIT:", vit)




#import timm

#print(timm.list_models())

#vit_timm = timm.create_model("vit_small_patch16_224")
#print(vit_timm)



network = Network(vit)

transform =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

dataset = CIFAR10("datasets", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=512, shuffle=True)
loss_func = nn.CrossEntropyLoss()

epochs = 20
optim = torch.optim.Adam(network.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs*len(loader), eta_min=1e-7)

for i in range(epochs):
    print("EPOCHS:", i)

    for x, y in tqdm(loader):
        # x: (batch_size, 3, 32, 32)

        # y: (batch_size,)
        preds = network(x) # (batch_size, 10)

        loss = loss_func(preds, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        pred_labels = torch.argmax(preds, dim=-1)
        acc = torch.sum(pred_labels == y) / len(y)
        print("LOSS:", loss.item(), "   ACC:", acc)