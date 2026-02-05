from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from Main.S1.ingest import WaveformEvents


@dataclass(frozen=True)
class ExportBundle:
    waves: Dict[int, WaveformEvents]


def split_by_class(X: np.ndarray, T0: np.ndarray, pred: np.ndarray) -> ExportBundle:
    if X.ndim != 3:
        raise ValueError(f"X must have shape (C, N, t). Got {X.shape}.")
    if T0.ndim != 2:
        raise ValueError(f"T0 must have shape (C, N). Got {T0.shape}.")
    if pred.ndim != 2:
        raise ValueError(f"pred must have shape (C, N). Got {pred.shape}.")
    if X.shape[:2] != T0.shape or T0.shape != pred.shape:
        raise ValueError("X, T0, and pred must align on (C, N).")

    classes = np.unique(pred[pred >= 0])
    waves: Dict[int, WaveformEvents] = {}

    for cls in classes:
        mask = pred == cls
        X_list = []
        T0_list = []
        for ch in range(X.shape[0]):
            idx = mask[ch]
            X_list.append(X[ch, idx, :])
            T0_list.append(T0[ch, idx])
        X_cls = _pad_channel_events(X_list)
        T0_cls = _pad_channel_events([arr[:, None] for arr in T0_list]).squeeze(-1)
        waves[int(cls)] = WaveformEvents(X=X_cls, T0=T0_cls)

    return ExportBundle(waves=waves)


def export_to_excel(bundle: ExportBundle, out_dir: str | Path) -> None:
    if pd is None:
        raise RuntimeError("pandas is required to export Excel files.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls, wave in bundle.waves.items():
        x_path = out_dir / f"class_{cls}_X.xlsx"
        t0_path = out_dir / f"class_{cls}_T0.xlsx"
        _write_wave_excel(wave.X, x_path)
        _write_wave_excel(wave.T0, t0_path, is_vector=True)


def _pad_channel_events(events: list[np.ndarray]) -> np.ndarray:
    max_n = max((arr.shape[0] for arr in events), default=0)
    if max_n == 0:
        return np.empty((len(events), 0, events[0].shape[1] if events else 0))

    time_len = events[0].shape[1]
    out = np.full((len(events), max_n, time_len), np.nan, dtype=float)
    for ch, arr in enumerate(events):
        out[ch, : arr.shape[0], :] = arr
    return out


def _write_wave_excel(data: np.ndarray, path: Path, is_vector: bool = False) -> None:
    with pd.ExcelWriter(path) as writer:
        for ch in range(data.shape[0]):
            sheet = f"ch_{ch:03d}"
            if is_vector:
                frame = pd.DataFrame(data[ch].reshape(1, -1))
            else:
                frame = pd.DataFrame(data[ch])
            frame.to_excel(writer, sheet_name=sheet, index=False, header=False)
