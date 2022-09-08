"""
An example of MoCo applied to the CIFAR10 dataset.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing import Optional

#from src import archs, models, validators
from src.archs import ResNet
from src.models.pretraining import MoCo
from src.models.finetuning import Classifier
from src.validators import TopKNN, Accuracy
from src import Trainer
from utils import load_cifar10, plot_results

def pretrain(epochs: int, data_root: str = "data", device: str = "cuda"):

    (train_bs, valid_bs, index_bs) = (512, 1024, 1024)

    dataloaders = {
        "train" : load_cifar10(data_root, train=True, batch_size=train_bs, shuffle=True),
        "valid" : load_cifar10(data_root, train=False, batch_size=valid_bs),
        "index" : load_cifar10(data_root, train=True, batch_size=index_bs),
    }

    validator = TopKNN(dataloaders, device=device)

    encoder = ResNet(channels_in=3, encode_dim=128, model_name="resnet-18")
    model = MoCo(encoder, queue_size=4096, momentum=0.99, temperature=0.1)

    optim = torch.optim.Adam(model.parameters(), lr=0.06, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optim, epochs * len(dataloaders["train"]))

    trainer = Trainer(optim, scheduler, validator)
    losses, valid_metrics = trainer.train(model, dataloaders["train"], epochs, device=device)

    plot_results(losses, epochs, title="Pretraining", y_str="Loss")
    plot_results(valid_metrics, epochs, title="Pretraining validation", y_str=validator.metric_str.capitalize(), color="orange")

    return model.encoder

def finetune(encoder: nn.Module, epochs: int, data_root: str = "data", load_path: Optional[str] = None, device: str = "cuda"):

    if load_path:
        encoder.load_state_dict(torch.load(load_path))

    (train_bs, valid_bs) = (512, 1024)

    dataloaders = {
        "train" : load_cifar10(data_root, train=True, batch_size=train_bs, shuffle=True),
        "valid" : load_cifar10(data_root, train=False, batch_size=valid_bs),
    }

    validator = Accuracy(dataloaders, device=device)

    model = Classifier(encoder, num_classes=10)

    optim = torch.optim.Adam(model.parameters(), lr=10, weight_decay=0)
    scheduler = CosineAnnealingLR(optim, epochs * len(dataloaders["train"]))

    trainer = Trainer(optim, scheduler, validator)
    losses, valid_metrics = trainer.train(model, dataloaders["train"], epochs, device=device)

    plot_results(losses, epochs, title="Finetuning", y_str="Loss")
    plot_results(valid_metrics, epochs, title="Finetuning validation", y_str=validator.metric_str.capitalize(), color="orange")

    return model.encoder

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_encoder = pretrain(epochs=10, device=device)
    finetuned_encoder = finetune(pretrained_encoder, epochs=10, device=device)

    return finetuned_encoder

if __name__ == "__main__":

    main()

    #device = "cuda" if torch.cuda.is_available() else "cpu"

    """transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])"""

    """data_root = "data"
    train_bs = 512
    valid_bs = 1024
    index_bs = 1024
    #train_dl = build_dataloader(CIFAR10, data_root, transform, batch_size=512, train=True, download=True)

    dataloaders = {
        "train" : load_cifar10(data_root, train=True, batch_size=train_bs, shuffle=True),
        "valid" : load_cifar10(data_root, train=False, batch_size=valid_bs),
        "index" : load_cifar10(data_root, train=True, batch_size=index_bs),
    }

    pretrain_validator = validators.TopKNN(dataloaders, device=device)

    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    encoder = archs.ResNet(3, encode_dim=128, model_name="resnet-18")
    pretrain_model = models.pretraining.MoCo(encoder, augment, queue_size=4096, momentum=0.99, temperature=0.1)

    epochs = 10
    pretrain_optim = torch.optim.Adam(pretrain_model.parameters(), lr=0.06, weight_decay=5e-4)
    pretrain_scheduler = CosineAnnealingLR(pretrain_optim, epochs * len(train_dl))

    #pretrainer = Trainer(pretrain_optim, pretrain_scheduler, pretrain_validator)

    main(pretrain_model, train_dl, epochs, pretrain_validator, pretrain_optim, pretrain_scheduler, device=device)

    #train_dl = build_dataloader(CIFAR10, data_root, transform, batch_size=512, train=True)
    #valid_dl = build_dataloader(CIFAR10, data_root, transform, batch_size=1024, train=False)
    #train_dl = load_cifar10(data_root, train=True, transform=transform, batch_size=512, download=False)
    #valid_dl = load_cifar10(data_root, train=False, transform=transform, batch_size=1024, download=False)

    dataloaders = {
        "train" : load_cifar10(data_root, train=True, transform=transform, batch_size=512, download=False),
        "valid" : load_cifar10(data_root, train=False, transform=transform, batch_size=1024, download=False),
    }

    finetune_validator = validators.Accuracy(valid_dl, device=device)

    finetune_model = models.finetuning.Classifier(encoder, num_classes=10)

    epochs = 10
    finetune_optim = torch.optim.Adam(finetune_model.parameters(), lr=30, weight_decay=0)
    finetune_scheduler = CosineAnnealingLR(finetune_optim, epochs * len(dataloaders["train"]))

    main(finetune_model, train_dl, epochs, finetune_validator, finetune_optim, finetune_scheduler, device=device)"""




"""def main(model, train_dl, epochs, validator, optim, scheduler=None, device="cuda"):

    trainer = Trainer(optim, scheduler, validator)
    losses, valid_metrics = trainer.train(model, train_dl, epochs, device=device)

    plot_results(losses, epochs, title="Training", y_str="Loss")
    plot_results(valid_metrics, epochs, title="Validation", y_str=validator.metric_str.capitalize(), color="orange")"""




"""model_dict = {
    "moco" : models.pretraining.MoCo,
}

def pretrain():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 0.06
    epochs = 5

    train_bs = 1024
    index_bs = 1024
    valid_bs = 1024

    weight_decay = 5e-4
    encode_dim = 128
    K = 4096
    m = 0.99
    temperature = 0.1
    # uses symmetric loss?

    augment = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ToTensor(),
        #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    train_ds = CIFAR10(root="static/datasets", train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, train_bs, shuffle=True)

    index_ds = CIFAR10(root="static/datasets", train=True, transform=transform)
    index_dl = DataLoader(index_ds, index_bs, shuffle=False)

    valid_ds = CIFAR10(root="static/datasets", train=False, download=True, transform=transform)
    valid_dl = DataLoader(valid_ds, valid_bs, shuffle=False)

    encoder = archs.ResNet(3, encode_dim, model_name="resnet-18")
    model = models.pretraining.MoCo(encoder, augment, K, m, temperature)

    #model.load_state_dict(torch.load("weights/"))

    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optim, epochs * len(train_dl))

    validator = validators.TopKNN({
        "index" : index_dl, "valid" : valid_dl,
    }, device=device)

    pretrainer = Trainer(optim, scheduler, validator)
    
    save_freq = 5
    valid_freq = 5
    losses, valid_metrics = pretrainer.train(model, train_dl, epochs, save_freq, valid_freq, device=device)
    #plot_results(losses, accs, valid_freq=5)
    plot_results(losses, epochs, title="Training", y_str="Loss")
    plot_results(valid_metrics, epochs, title="Validation", y_str=validator.metric_str.capitalize(), color="orange")
    return model

def finetune(encoder):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 30
    epochs = 5
    train_bs = 512
    valid_bs = 1024

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    train_ds = CIFAR10(root="static/datasets", train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, train_bs, shuffle=True)

    valid_ds = CIFAR10(root="static/datasets", train=False, download=True, transform=transform)
    valid_dl = DataLoader(valid_ds, valid_bs, shuffle=False)

    model = models.finetuning.Classifier(encoder, num_classes=10)

    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=0)
    scheduler = CosineAnnealingLR(optim, epochs * len(train_dl))

    finetuner = Trainer(optim, scheduler)

    validator = validators.Accuracy(valid_dl, device=device)

    losses, valid_metrics = finetuner.train(model, train_dl, epochs, validator, save_freq=5, valid_freq=5, device=device)
    plot_results(losses, accs, valid_freq=5)

def main():

    model = pretrain()
    finetune(model.encoder)"""