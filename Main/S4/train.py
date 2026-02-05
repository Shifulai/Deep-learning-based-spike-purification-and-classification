from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

import h5py
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None



@dataclass(frozen=True)
class TrainData:
    X: np.ndarray
    y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


class WaveformDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise RuntimeError("PyTorch is required for WaveformDataset.")
        if X.ndim != 2:
            raise ValueError(f"X must have shape (N, t). Got {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must have shape (N,). Got {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_training_data(path: str | Path) -> TrainData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    if path.is_dir():
        raise ValueError("Please pass a .mat file path directly (directory input is not supported).")

    if path.suffix.lower() != ".mat":
        raise ValueError("Training data must be a .mat file containing Train_X and Train_Y.")

    with h5py.File(path, "r") as mat:
        if "Train_X" not in mat or "Train_Y" not in mat:
            raise ValueError("Training .mat must contain datasets 'Train_X' and 'Train_Y'.")
        X = np.array(mat["Train_X"], dtype=float).T
        y = np.array(mat["Train_Y"], dtype=float).reshape(-1)
        y = y.astype(int)

    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, t). Got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (N,). Got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    return TrainData(X=X, y=y)




def build_simple_mlp(input_dim: int, num_classes: int) -> "nn.Module":
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )



class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=[1,3], padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=[1,3], padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk





def Resnet(num_classes: int) -> "nn.Module":
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=[1,3], padding=1),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=[1,3], padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    return nn.Sequential(b1, b2, b3, b4,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(256, num_classes))





def train_model(
    data: TrainData,
    model: Optional["nn.Module"] = None,
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    plot_history: bool = False,
    device: Optional[str] = None,
) -> "nn.Module":
    if torch is None:
        raise RuntimeError("PyTorch is required for training.")

    num_classes = int(np.max(data.y)) + 1
    model = model or build_simple_mlp(data.X.shape[1], num_classes)
    expects_4d = _model_expects_4d(model)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = WaveformDataset(data.X, data.y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history_loss: list[float] = []
    history_acc: list[float] = []

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            if expects_4d:
                X_batch = X_batch[:, None, None, :]
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y_batch.size(0)
            total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
            total_samples += int(y_batch.size(0))

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = total_correct / max(1, total_samples)
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)

    if plot_history:
        if plt is None:
            raise RuntimeError("matplotlib is required to plot training history.")
        fig, ax1 = plt.subplots(figsize=(7, 3.5))
        ax1.plot(history_loss, label="loss", color="tab:red")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.plot(history_acc, label="acc", color="tab:blue")
        ax2.set_ylabel("acc", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        fig.tight_layout()
        plt.show()

    return model


def _model_expects_4d(model: "nn.Module") -> bool:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return True
    return False


def save_model(model: "nn.Module", path: str | Path) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to save a model.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
