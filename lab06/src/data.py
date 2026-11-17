"""
Lab 06: Datasets and data loaders

Fashion-MNIST with the same 55k / 5k split, seed and normalisation as
Lab 03. The 'fake' dataset is random images for the tests only.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import resolve_path


MEAN = 0.2860
STD = 0.3530


def make_datasets(cfg):
    data = cfg["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])

    if data["dataset"] == "fashion_mnist":
        root = resolve_path(data["root"])
        full_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    elif data["dataset"] == "fake":
        size = data["train_size"] + data["val_size"]
        full_train = datasets.FakeData(size=size, image_size=(1, 28, 28), num_classes=10,
                                       transform=transform)
        test_set = datasets.FakeData(size=data["val_size"], image_size=(1, 28, 28), num_classes=10,
                                     transform=transform, random_offset=1_000_000)
    else:
        raise ValueError(f"Unknown dataset {data['dataset']!r}")

    split_generator = torch.Generator().manual_seed(cfg["experiment"]["seed"])
    train_set, val_set = random_split(full_train, [data["train_size"], data["val_size"]],
                                      generator=split_generator)
    return train_set, val_set, test_set


def make_loaders(cfg, train_set, val_set, test_set):
    """Return the loaders and the generator that decides the shuffle order.

    The generator is saved in checkpoints, so a resumed run sees the same
    batches as an uninterrupted one. Only the training loader uses worker
    processes, because the worker study is about the training input pipeline.
    """

    data = cfg["data"]
    workers = data["num_workers"]
    generator = torch.Generator().manual_seed(cfg["experiment"]["seed"])

    train_loader = DataLoader(
        train_set,
        batch_size=data["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=data["eval_batch_size"])
    test_loader = DataLoader(test_set, batch_size=data["eval_batch_size"])

    return train_loader, val_loader, test_loader, generator
