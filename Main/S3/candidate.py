from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except Exception:  # pragma: no cover - optional dependency
    plt = None
    Slider = None

@dataclass(frozen=True)
class CandidateResult:
    X: np.ndarray
    T0: np.ndarray
    counts: list[int]
    threshold: Sequence[float] | float
    pre: int
    post: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.X.shape


def plot_waveform_overlay(
    X: np.ndarray,
    channels: Sequence[int] | None = None,
    max_waveforms: int = 200,
    title: str | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")
    if X.ndim != 3:
        raise ValueError(f"X must have shape (C, N, t). Got {X.shape}.")

    total_channels = X.shape[0]
    if channels is None:
        channels = [0]
    channels = [_normalize_channel_index(ch, total_channels) for ch in channels]
    channels = list(dict.fromkeys(channels))

    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(10, 2.2 * len(channels)))
    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        waveforms = X[ch]
        valid = ~np.isnan(waveforms).any(axis=1)
        waveforms = waveforms[valid]
        if waveforms.size == 0:
            ax.set_title(f"ch {ch} (no waveforms)")
            continue

        if waveforms.shape[0] > max_waveforms:
            waveforms = waveforms[:max_waveforms]

        mean_wave = np.mean(waveforms, axis=0)
        ax.plot(waveforms.T, color="steelblue", alpha=0.2, linewidth=0.7)
        ax.plot(mean_wave, color="crimson", linewidth=1.6)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    plt.show()


def pick_threshold(
    hp_signal: np.ndarray,
    channels: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
    initial: float | None = None,
) -> list[float]:
    if plt is None or Slider is None:
        raise RuntimeError("matplotlib is required for threshold visualization.")
    if hp_signal.ndim != 3:
        raise ValueError(f"hp_signal must have shape (C, 1, T). Got {hp_signal.shape}.")

    total_channels, _, total_len = hp_signal.shape
    if channels is None:
        channels = [0]
    channels = [_normalize_channel_index(ch, total_channels) for ch in channels]
    channels = list(dict.fromkeys(channels))

    start = 0 if start is None else max(start, 0)
    stop = total_len if stop is None else min(stop, total_len)
    if start >= stop:
        raise ValueError(f"Invalid range: start={start}, stop={stop}")

    segments = [hp_signal[ch, 0, start:stop] for ch in channels]
    if initial is None:
        window_min = float(min(np.min(seg) for seg in segments))
        initial = window_min * 1.5
    min_val = float(min(np.min(seg) for seg in segments))
    max_val = float(max(np.max(seg) for seg in segments))

    fig, axes = plt.subplots(len(channels), 1, sharex=True, figsize=(10, 2.2 * len(channels)))
    if len(channels) == 1:
        axes = [axes]
    bottom_pad = 0.12 + 0.06 * len(channels)
    plt.subplots_adjust(bottom=bottom_pad)

    threshold_lines = []
    for ax, ch, segment in zip(axes, channels, segments):
        ax.plot(segment, linewidth=0.8)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(True, alpha=0.3)
        threshold_lines.append(ax.axhline(-abs(initial), color="red", linewidth=1.2))

    axes[0].set_title("Threshold selection (displayed channels)")
    axes[-1].set_xlabel("Sample index")

    sliders = []
    slider_start = 0.08
    slider_height = 0.03
    slider_gap = 0.01
    for idx, ch in enumerate(channels):
        slider_ax = fig.add_axes([0.15, slider_start + idx * (slider_height + slider_gap), 0.7, slider_height])
        slider = Slider(
            ax=slider_ax,
            label=f"ch {ch}",
            valmin=min_val,
            valmax=max_val,
            valinit=initial,
        )
        sliders.append(slider)

    def _update(_: float) -> None:
        for line, slider in zip(threshold_lines, sliders):
            y = -abs(slider.val)
            line.set_ydata([y, y])
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(_update)
    plt.show()

    selected_vals = [float(slider.val) for slider in sliders]
    if selected_vals:
        avg_val = float(np.mean(selected_vals))
    else:
        avg_val = float(initial)
    thresholds = [avg_val for _ in range(total_channels)]
    for ch, val in zip(channels, selected_vals):
        thresholds[ch] = val
    return thresholds


def detect_candidates(
    hp_signal: np.ndarray,
    pre: int,
    post: int,
    threshold: Sequence[float] | float | None = None,
) -> CandidateResult:
    if hp_signal.ndim != 3:
        raise ValueError(f"hp_signal must have shape (C, 1, T). Got {hp_signal.shape}.")
    if pre < 0 or post < 0:
        raise ValueError("pre/post must be non-negative.")

    channels, _, total_len = hp_signal.shape
    if threshold is None:
        threshold = _default_threshold(hp_signal)
    thresholds = _expand_thresholds(threshold, channels)
    window = pre + post + 1

    channel_events: list[list[np.ndarray]] = []
    channel_t0: list[list[int]] = []

    for ch in range(channels):
        signal = hp_signal[ch, 0, :]
        thr = thresholds[ch]
        crossings = _find_negative_crossings(signal, thr)
        events: list[np.ndarray] = []
        t0_list: list[int] = []

        for t1 in crossings:
            start = t1 - pre
            end = t1 + post + 1
            if start < 0 or end > total_len:
                continue
            events.append(signal[start:end])
            t0_list.append(t1)

        channel_events.append(events)
        channel_t0.append(t0_list)

    max_n = max((len(ev) for ev in channel_events), default=0)
    if max_n == 0:
        X = np.empty((channels, 0, window), dtype=float)
        T0 = np.empty((channels, 0), dtype=int)
    else:
        X = np.full((channels, max_n, window), np.nan, dtype=float)
        T0 = np.full((channels, max_n), -1, dtype=int)
        for ch in range(channels):
            for idx, ev in enumerate(channel_events[ch]):
                X[ch, idx, :] = ev
            for idx, t0 in enumerate(channel_t0[ch]):
                T0[ch, idx] = t0

    counts = [len(ev) for ev in channel_events]
    return CandidateResult(X=X, T0=T0, counts=counts, threshold=threshold, pre=pre, post=post)


def _expand_thresholds(threshold: Sequence[float] | float, channels: int) -> list[float]:
    if isinstance(threshold, (float, int)):
        return [float(threshold)] * channels
    if len(threshold) != channels:
        raise ValueError("threshold length must match number of channels.")
    return [float(v) for v in threshold]


def _find_negative_crossings(signal: np.ndarray, threshold: float) -> list[int]:
    thr = -abs(threshold)
    below = signal < thr
    crossings = []
    for idx in range(1, len(signal)):
        if not below[idx - 1] and below[idx]:
            crossings.append(idx)
    return crossings


def _default_threshold(hp_signal: np.ndarray) -> float:
    min_val = float(np.min(hp_signal))
    return min_val * 0.5


def _normalize_channel_index(index: int, size: int) -> int:
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise IndexError(f"Channel index out of range: {index}")
    return index
