import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Callable

def load_cifar10(
    root: str,
    train: bool,
    batch_size: int,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
    download: bool = True
):
    if transform is None:
        transform =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

    dataset = CIFAR10(root, train, transform, download=download)
    return DataLoader(dataset, batch_size, shuffle=shuffle)

def plot_results(
    metrics: list,
    epochs: int,
    title: str,
    y_str: str,
    x_str: str = "Epochs",
    color="blue",
    path="static/plots",
):
    plt.title(title)
    plt.xlabel(x_str)
    plt.ylabel(y_str)

    freq = epochs // len(metrics)
    x = [i * freq for i in range(1, len(metrics)+1)]

    plt.plot(x, metrics, color=color)
    plt.savefig(f"{path}/{title.lower()}.png")
    plt.show()