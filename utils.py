import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Callable

"""def plot_results(losses, accs, valid_freq):

    plt.title("Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    x = [i for i in range(1, len(losses)+1)]
    plt.plot(x, losses)
    plt.savefig("static/plots/training.png")
    plt.show()

    plt.title("Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    x = [valid_freq*i for i in range(1, len(accs)+1)]
    plt.plot(x, accs, color="orange")
    plt.savefig("static/plots/validation.png")
    plt.show()"""

"""train_ds = CIFAR10(root="static/datasets", train=True, download=True, transform=transform)
train_dl = DataLoader(train_ds, train_bs, shuffle=True)

index_ds = CIFAR10(root="static/datasets", train=True, transform=transform)
index_dl = DataLoader(index_ds, index_bs, shuffle=False)

valid_ds = CIFAR10(root="static/datasets", train=False, download=True, transform=transform)
valid_dl = DataLoader(valid_ds, valid_bs, shuffle=False)"""

"""def build_dataloader(dataset_cls: Type[VisionDataset], root: str, transform: Callable, batch_size: int, shuffle: bool = True, **dataset_kwargs):

    dataset = dataset_cls(root, transform=transform, **dataset_kwargs)
    return DataLoader(dataset, batch_size, shuffle)"""

"""from torchvision.datasets import CIFAR10
batch_size = 16
build_dataloader(CIFAR10, "static/datasets", None, batch_size, train=True, download=True)"""

def load_cifar10(root: str, train: bool, batch_size: int, transform: Optional[Callable] = None, shuffle: bool = False, download: bool = True):

    if transform is None:
        transform =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

    dataset = CIFAR10(root, train, transform, download=download)
    return DataLoader(dataset, batch_size, shuffle=shuffle)

def plot_results(metrics: list, epochs: int, title: str, y_str: str, x_str: str = "Epochs", color="blue"):

    plt.title(title)
    plt.xlabel(x_str)
    plt.ylabel(y_str)

    freq = epochs // len(metrics)
    x = [i * freq for i in range(1, len(metrics)+1)]

    plt.plot(x, metrics, color=color)
    plt.savefig(f"plots/{title.lower()}.png")
    plt.show()