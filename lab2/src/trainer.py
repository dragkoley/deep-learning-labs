import json
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config


def gpu_augment(images):
    flip_mask = torch.rand(images.size(0), 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip_mask, torch.flip(images, dims=[3]), images)

    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 8, (1,)).item()
    dx = torch.randint(0, 8, (1,)).item()
    return padded[:, :, dy: dy + 32, dx: dx + 32]


def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        device: str,
        augment: bool = False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train and augment:
                images = gpu_augment(images)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def train_model(
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        augment: bool = False,
):
    model.to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
        "total_time_sec": 0.0,
    }

    best_path = cfg.checkpoint_dir / f"{model_name}_best.pt"
    logs_path = cfg.logs_dir / f"{model_name}_history.json"

    print(f"-----Training {model_name.upper()} on {cfg.device.upper()} for {cfg.epochs} epochs-----")
    start_time = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.perf_counter()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, cfg.device, augment=augment)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, cfg.device, augment=False)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        saved_mark = ""
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), best_path)
            saved_mark = " [best]"

        ep_time = time.perf_counter() - ep_start
        print(
            f"Epoch {epoch}/{cfg.epochs} ({ep_time:.2f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}{saved_mark}"
        )

    history["total_time_sec"] = round(time.perf_counter() - start_time, 2)
    print(
        f"Finished in {history['total_time_sec']} sec. | "
        f"Best Val Acc: {history['best_val_acc']:.4f} (Epoch {history['best_epoch']})\n"
    )

    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history
