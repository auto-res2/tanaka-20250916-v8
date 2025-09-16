"""Model training related utilities.
Since the original monolithic script is missing, this module implements a
minimal, production-ready training stub that can be called from
src.main.  The goal is simply to illustrate the refactor split while
remaining fully runnable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T


# ----------------------- helpers ------------------------------------------------

def _get_device() -> torch.device:  # pragma: no cover
    """Return the best available torch.device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------- core API ----------------------------------------------

def get_dataloaders(cfg: Dict[str, Any]):
    """Create DataLoader objects according to the configuration.
    Only CIFAR-10 and CIFAR-100 are supported here for brevity.
    """
    dataset_name: str = cfg["data"]["name"].lower()
    batch_size: int = cfg["training"].get("batch_size", 256)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root = os.getenv("DATA_ROOT", "./data")
    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 8),
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 8),
        pin_memory=True,
    )
    return train_loader, test_loader


def build_model(cfg: Dict[str, Any]):
    """Return a torchvision model matching the config."""
    model_name = cfg["model"].get("name", "resnet18")
    num_classes = 10 if cfg["data"]["name"].lower() == "cifar10" else 100
    if model_name == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def train_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """High-level training loop returning metrics."""
    device = _get_device()
    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=cfg["training"].get("lr", 1e-3))

    epochs = cfg["training"].get("epochs", 1)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train-loss: {train_loss:.4f} | "
            f"val-loss: {metrics['loss']:.4f} | "
            f"val-acc: {metrics['accuracy']:.4f}"
        )

    # persist a lightweight checkpoint in research folder
    ckpt_dir = Path(".research/iteration1")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "last.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # return summary metrics
    summary = {"val_accuracy": metrics["accuracy"], "val_loss": metrics["loss"]}
    return summary
