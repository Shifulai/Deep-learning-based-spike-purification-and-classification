from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


import matplotlib.pyplot as plt



from scipy.signal import butter, filtfilt


from Main.S1.ingest import RawSignal


@dataclass(frozen=True)
class FilterResult:
    data: np.ndarray
    cutoff_hz: float | tuple[float, float]
    fs_hz: float
    method: str
    order: int


def select_channels(raw: RawSignal, channels: Sequence[int | str] | None) -> tuple[np.ndarray, list[str]]:
    if channels is None:
        return raw.data, raw.channel_names

    indices: list[int] = []
    for item in channels:
        if isinstance(item, int):
            indices.append(_normalize_index(item, len(raw.channel_names)))
        else:
            if item not in raw.channel_names:
                raise ValueError(f"Unknown channel name: {item}")
            indices.append(raw.channel_names.index(item))

    indices = _unique_ordered(indices)
    data = raw.data[indices, :, :]
    names = [raw.channel_names[idx] for idx in indices]
    return data, names


def plot_raw(
    raw: RawSignal,
    channels: Sequence[int | str] | None = None,
    start: int | None = None,
    stop: int | None = None,
    max_channels: int = 8,
    title: str | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")

    data, names = select_channels(raw, channels)
    if data.shape[0] > max_channels:
        data = data[:max_channels]
        names = names[:max_channels]

    start = 0 if start is None else max(start, 0)
    stop = data.shape[2] if stop is None else min(stop, data.shape[2])
    if start >= stop:
        raise ValueError(f"Invalid range: start={start}, stop={stop}")

    segment = data[:, 0, start:stop]
    fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=(10, 2.2 * len(names)))
    if len(names) == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.plot(segment[idx], linewidth=0.8)
        ax.set_ylabel(names[idx])
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    plt.show()


def plot_hp(
    hp: np.ndarray | FilterResult,
    channels: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
    max_channels: int = 8,
    title: str | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")

    data = hp.data if isinstance(hp, FilterResult) else hp
    if data.ndim != 3:
        raise ValueError(f"hp data must have shape (C, 1, T). Got {data.shape}.")

    total_channels = data.shape[0]
    if channels is None:
        indices = list(range(total_channels))
    else:
        indices = [_normalize_index(ch, total_channels) for ch in channels]

    indices = list(dict.fromkeys(indices))
    if len(indices) > max_channels:
        indices = indices[:max_channels]

    start = 0 if start is None else max(start, 0)
    stop = data.shape[2] if stop is None else min(stop, data.shape[2])
    if start >= stop:
        raise ValueError(f"Invalid range: start={start}, stop={stop}")

    segment = data[indices, 0, start:stop]
    fig, axes = plt.subplots(len(indices), 1, sharex=True, figsize=(10, 2.2 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.plot(segment[idx], linewidth=0.8)
        ax.set_ylabel(f"ch {indices[idx]}")
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    plt.show()


def highpass_filter(
    raw: RawSignal,
    fs_hz: float,
    cutoff_hz: float | tuple[float, float] = 200.0,
    order: int = 4,
    method: str = "butter",
) -> FilterResult:
    if butter is None or filtfilt is None:
        raise RuntimeError("scipy is required for filtering.")

    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")

    nyquist = fs_hz / 2.0
    if isinstance(cutoff_hz, tuple):
        low, high = cutoff_hz
        if not (0 < low < high < nyquist):
            raise ValueError("cutoff_hz must satisfy 0 < low < high < fs/2.")
        btype = "bandpass"
        wn = [low / nyquist, high / nyquist]
    else:
        if not (0 < cutoff_hz < nyquist):
            raise ValueError("cutoff_hz must satisfy 0 < cutoff < fs/2.")
        btype = "highpass"
        wn = cutoff_hz / nyquist

    if method != "butter":
        raise ValueError(f"Unsupported method: {method}")

    b, a = butter(order, wn, btype=btype)
    data = filtfilt(b, a, raw.data, axis=2)
    return FilterResult(data=data, cutoff_hz=cutoff_hz, fs_hz=fs_hz, method=method, order=order)


def _normalize_index(index: int, size: int) -> int:
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise IndexError(f"Channel index out of range: {index}")
    return index


def _unique_ordered(items: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
