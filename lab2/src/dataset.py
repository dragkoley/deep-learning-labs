import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from src.config import Config

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _to_gpu_tensors(raw_dataset: datasets.CIFAR10, device: str) -> tuple[Tensor, Tensor]:
    images = torch.from_numpy(raw_dataset.data).to(device)
    images = images.permute(0, 3, 1, 2).float()
    images = (images / 255.0 - 0.5) / 0.5
    labels = torch.tensor(raw_dataset.targets, dtype=torch.long).to(device)
    return images, labels


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    full_train = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True)

    train_x, train_y = _to_gpu_tensors(full_train, cfg.device)
    test_x, test_y = _to_gpu_tensors(test_dataset, cfg.device)

    indices = list(range(len(train_y)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.val_split, random_state=cfg.seed, stratify=full_train.targets
    )

    train_dataset = TensorDataset(train_x[train_idx], train_y[train_idx])
    val_dataset = TensorDataset(train_x[val_idx], train_y[val_idx])
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    return train_loader, val_loader, test_loader
