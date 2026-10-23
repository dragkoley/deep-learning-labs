from dataclasses import dataclass
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    data_dir: Path = LAB_DIR / "data"
    checkpoint_dir: Path = LAB_DIR / "checkpoints"
    logs_dir: Path = LAB_DIR / "logs"

    num_classes: int = 10
    img_size: int = 32
    val_split: float = 0.1

    seed: int = 100
    batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)