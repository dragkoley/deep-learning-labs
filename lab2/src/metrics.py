import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import CIFAR10_CLASSES


def load_logs(cfg: Config, filename: str) -> dict:
    logs_path = cfg.logs_dir / filename
    with open(logs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_losses(cfg: Config, vgg_file: str, resnet_file: str, title: str) -> None:
    vgg_hist = load_logs(cfg, vgg_file)
    resnet_hist = load_logs(cfg, resnet_file)

    epochs_range = range(1, len(vgg_hist["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)

    models_data = [
        ("VGG", vgg_hist, axes[0, 0], axes[0, 1]),
        ("ResNet", resnet_hist, axes[1, 0], axes[1, 1]),
    ]

    for name, hist, ax_loss, ax_acc in models_data:
        best_ep = hist["best_epoch"]

        ax_loss.plot(epochs_range, hist["train_loss"], label="Train Loss", color="blue", lw=2)
        ax_loss.plot(epochs_range, hist["val_loss"], label="Val Loss", color="red", lw=2)
        ax_loss.axvline(best_ep, color="gray", linestyle="--", label=f"Best Epoch ({best_ep})")
        ax_loss.set_title(f"{name} — Loss per Epoch")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("CrossEntropy Loss")
        ax_loss.legend()
        ax_loss.grid(True, linestyle="--", alpha=0.5)

        ax_acc.plot(epochs_range, hist["train_acc"], label="Train Acc", color="blue", lw=2)
        ax_acc.plot(epochs_range, hist["val_acc"], label="Val Acc", color="red", lw=2)
        ax_acc.axvline(best_ep, color="gray", linestyle="--", label=f"Best Acc ({hist['best_val_acc']:.4f})")
        ax_acc.set_title(f"{name} — Accuracy per Epoch")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_checkpoint(
        model: nn.Module, model_name: str, display_name: str, test_loader: DataLoader, cfg: Config
) -> tuple[dict, list[int], list[int]]:
    ckpt_path = cfg.checkpoint_dir / f"{model_name}_best.pt"
    log_data = load_logs(cfg, f"{model_name}_history.json")

    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
    model.to(cfg.device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(cfg.device)
            output = model(images)
            preds = output.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    metrics_row = {
        "Model": display_name,
        "Accuracy": round(acc, 4),
        "Precision (macro)": round(prec, 4),
        "Recall (macro)": round(rec, 4),
        "F1-score (macro)": round(f1, 4),
        "Parameters": f"{count_parameters(model):,}",
        "Train time": log_data["total_time_sec"],
        "Best epoch": log_data["best_epoch"]
    }

    return metrics_row, all_labels, all_preds


def plot_confusion_matrices(results_list: list[tuple[str, list[int], list[int]]]) -> None:
    fig, axes = plt.subplots(1, len(results_list), figsize=(15, 10))
    fig.suptitle("Confusion Matrices on Test Dataset")

    for ax, (name, y_true, y_pred) in zip(axes, results_list):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
        ax.set_title(name)
        ax.grid(False)


plt.tight_layout()
plt.show()
