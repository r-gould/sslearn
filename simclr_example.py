"""
An example of MoCo applied to the CIFAR10 dataset.
"""

import torch
import torch.nn as nn

from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR

#from src import archs, models, validators
from sslearn.archs import ResNet
from sslearn.models.pretraining import SimCLR
from sslearn.models.finetuning import Classifier
from sslearn.validators import TopKNN, Accuracy
from sslearn.schedulers import CosineAnnealingLinearWarmup
from sslearn import Trainer
from utils import load_cifar10, plot_results

def pretrain(epochs: int, warmup_epochs: int, data_root: str, device: str = "cuda"):

    (train_bs, valid_bs, index_bs) = (512, 1024, 1024)

    dataloaders = {
        "train" : load_cifar10(data_root, train=True, batch_size=train_bs, shuffle=True),
        "valid" : load_cifar10(data_root, train=False, batch_size=valid_bs),
        "index" : load_cifar10(data_root, train=True, batch_size=index_bs),
    }

    validator = TopKNN(dataloaders, device=device)

    encoder = ResNet(channels_in=3, model_name="resnet-18")
    model = SimCLR(encoder, head_dim=encoder.encode_dim, temperature=0.1)

    #lr = 0.3 * (train_bs / 256)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

    num_batches = len(dataloaders["train"])
    warmup_steps = warmup_epochs * num_batches
    total_steps = epochs * num_batches
    scheduler = CosineAnnealingLinearWarmup(optim, warmup_steps, total_steps)

    trainer = Trainer(optim, scheduler, validator)
    losses, valid_metrics = trainer.train(model, dataloaders["train"], epochs, device=device)

    plot_results(losses, epochs, title="Pretraining", y_str="Loss")
    plot_results(valid_metrics, epochs, title="Pretraining validation", y_str=validator.metric_str.capitalize(), color="orange")

    return model.encoder

def finetune(encoder: nn.Module, epochs: int, data_root: str, load_path: Optional[str] = None, device: str = "cuda"):

    if load_path:
        encoder.load_state_dict(torch.load(load_path))

    (train_bs, valid_bs) = (512, 1024)

    dataloaders = {
        "train" : load_cifar10(data_root, train=True, batch_size=train_bs, shuffle=True),
        "valid" : load_cifar10(data_root, train=False, batch_size=valid_bs),
    }

    validator = Accuracy(dataloaders, device=device)

    model = Classifier(encoder, num_classes=10)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)
    scheduler = CosineAnnealingLR(optim, epochs * len(dataloaders["train"]))

    trainer = Trainer(optim, scheduler, validator)
    losses, valid_metrics = trainer.train(model, dataloaders["train"], epochs, device=device)

    plot_results(losses, epochs, title="Finetuning", y_str="Loss")
    plot_results(valid_metrics, epochs, title="Finetuning validation", y_str=validator.metric_str.capitalize(), color="orange")

    return model.encoder

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "static/datasets"

    pretrained_encoder = pretrain(epochs=10, data_root=data_root, device=device)
    finetuned_encoder = finetune(pretrained_encoder, epochs=10, data_root=data_root, device=device)

    return finetuned_encoder

if __name__ == "__main__":

    main()