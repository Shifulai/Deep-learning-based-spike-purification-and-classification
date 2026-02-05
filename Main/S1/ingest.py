from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np


import pandas as pd



import h5py




@dataclass(frozen=True)
class ChannelSeries:
    name: str
    values: np.ndarray
    source: Path


@dataclass(frozen=True)
class RawSignal:
    data: np.ndarray
    channel_names: list[str]
    sources: list[Path]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape


@dataclass(frozen=True)
class WaveformEvents:
    X: np.ndarray
    T0: np.ndarray
    channel_names: Optional[list[str]] = None
    sources: list[Path] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.X.shape


@dataclass(frozen=True)
class SpikeInput:
    kind: Literal["raw", "wave"]
    raw: Optional[RawSignal] = None
    wave: Optional[WaveformEvents] = None


def load_input(input_path: str | Path, data_type: Literal["raw", "wave"]) -> SpikeInput:
    path = Path(input_path)
    if data_type == "raw":
        return SpikeInput(kind="raw", raw=load_raw(path))
    if data_type == "wave":
        return SpikeInput(kind="wave", wave=load_wave(path))
    raise ValueError(f"Unsupported data_type: {data_type}")


def load_raw(input_path: Path) -> RawSignal:
    paths = _collect_input_files(input_path, (".mat",))
    channels: list[ChannelSeries] = []

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".mat":
            channels.extend(_read_mat_channels(path))
            continue
        raise ValueError(f"Unsupported file type: {path}")

    if not channels:
        raise FileNotFoundError(f"No input files found under {input_path}")

    lengths = {len(ch.values) for ch in channels}
    if len(lengths) != 1:
        raise ValueError(f"Channel lengths mismatch: {sorted(lengths)}")

    data = np.stack([ch.values for ch in channels], axis=0)
    data = data[:, None, :]
    return RawSignal(
        data=data,
        channel_names=[ch.name for ch in channels],
        sources=[ch.source for ch in channels],
    )


def load_wave(input_path: Path) -> WaveformEvents:
    pair = _find_wave_pair(input_path)
    if pair is None:
        raise FileNotFoundError(
            "Waveform input requires two Excel files (X and T0). "
            "Place them in the same folder or pass one of the files directly."
        )

    x_path, t0_path = pair
    X, x_names = _read_wave_X(x_path)
    T0, t0_names = _read_wave_T0(t0_path)
    _validate_wave_shapes(X, T0)
    channel_names = _merge_channel_names(x_names, t0_names, X.shape[0])
    return WaveformEvents(
        X=X,
        T0=T0,
        channel_names=channel_names,
        sources=[x_path, t0_path],
    )


def _collect_input_files(input_path: Path, suffixes: Sequence[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = [p for p in input_path.iterdir() if p.suffix.lower() in suffixes]
    return sorted(files)


def _find_wave_pair(input_path: Path) -> Optional[tuple[Path, Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        files = [input_path]
    else:
        files = [p for p in input_path.iterdir() if p.suffix.lower() in (".xlsx", ".xls")]
    if not files:
        return None

    x_path = _pick_named_file(files, {"x", "wave_x", "waveform_x"})
    t0_path = _pick_named_file(files, {"t0", "t_0", "wave_t0", "waveform_t0"})

    if x_path is None:
        x_candidates = [p for p in files if p.stem.lower().startswith("x")]
        if len(x_candidates) == 1:
            x_path = x_candidates[0]
        elif len(x_candidates) > 1:
            names = ", ".join(p.name for p in x_candidates)
            raise ValueError(f"Multiple X files found; rename to X.*: {names}")

    if t0_path is None:
        t0_candidates = [p for p in files if p.stem.lower().startswith("t0")]
        if len(t0_candidates) == 1:
            t0_path = t0_candidates[0]
        elif len(t0_candidates) > 1:
            names = ", ".join(p.name for p in t0_candidates)
            raise ValueError(f"Multiple T0 files found; rename to T0.*: {names}")

    if x_path is None or t0_path is None:
        return None
    if x_path == t0_path:
        raise ValueError("X and T0 cannot be the same file.")
    return x_path, t0_path


def _pick_named_file(files: Sequence[Path], stems: set[str]) -> Optional[Path]:
    for path in files:
        if path.stem.lower() in stems:
            return path
    return None


def _read_wave_X(path: Path) -> tuple[np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        if pd is None:
            raise RuntimeError(
                "Excel input requires pandas + openpyxl. "
                "Please install them to read Excel files."
            )
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        channels: list[np.ndarray] = []
        channel_names: list[str] = []
        for sheet_name, frame in sheets.items():
            values = frame.to_numpy(dtype=float)
            if values.ndim != 2:
                raise ValueError(f"X sheet {sheet_name} must be 2D. Got {values.shape}.")
            channels.append(values)
            channel_names.append(sheet_name)
        _ensure_same_shape(channels, "X")
        return np.stack(channels, axis=0), channel_names

    raise ValueError(f"Unsupported X file type: {path}")


def _read_wave_T0(path: Path) -> tuple[np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        if pd is None:
            raise RuntimeError(
                "Excel input requires pandas + openpyxl. "
                "Please install them to read Excel files."
            )
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        channels: list[np.ndarray] = []
        channel_names: list[str] = []
        for sheet_name, frame in sheets.items():
            values = frame.to_numpy(dtype=float).reshape(-1)
            channels.append(values)
            channel_names.append(sheet_name)
        _ensure_same_length(channels, "T0")
        return np.stack(channels, axis=0), channel_names

    raise ValueError(f"Unsupported T0 file type: {path}")


def _ensure_same_shape(channels: Sequence[np.ndarray], label: str) -> None:
    shapes = {ch.shape for ch in channels}
    if len(shapes) != 1:
        raise ValueError(f"{label} channel shapes mismatch: {sorted(shapes)}")


def _ensure_same_length(channels: Sequence[np.ndarray], label: str) -> None:
    lengths = {len(ch) for ch in channels}
    if len(lengths) != 1:
        raise ValueError(f"{label} channel lengths mismatch: {sorted(lengths)}")


def _validate_wave_shapes(X: np.ndarray, T0: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must have shape (C, N, t). Got {X.shape}.")
    if T0.ndim != 2:
        raise ValueError(f"T0 must have shape (C, N). Got {T0.shape}.")
    if X.shape[0] != T0.shape[0] or X.shape[1] != T0.shape[1]:
        raise ValueError(
            f"X and T0 shape mismatch: X={X.shape}, T0={T0.shape} (expected C,N aligned)"
        )


def _merge_channel_names(
    x_names: Optional[Sequence[str]],
    t0_names: Optional[Sequence[str]],
    count: int,
) -> list[str]:
    if x_names and t0_names and list(x_names) != list(t0_names):
        raise ValueError("X and T0 channel names do not match.")
    if x_names:
        return list(x_names)
    if t0_names:
        return list(t0_names)
    return [f"ch_{idx:03d}" for idx in range(count)]


def _read_excel_channels(path: Path) -> Iterable[ChannelSeries]:
    assert pd is not None
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    for sheet_name, frame in sheets.items():
        values = frame.to_numpy(dtype=float).reshape(-1)
        name = f"{path.stem}:{sheet_name}"
        yield ChannelSeries(name=name, values=values, source=path)


def _read_mat_channels(path: Path) -> Iterable[ChannelSeries]:
    if h5py is None:
        raise RuntimeError("h5py is required to read .mat files.")
    with h5py.File(path, "r") as mat:
        data = _extract_mat_matrix(mat)
        if data.ndim != 2:
            raise ValueError(f"MAT data must be 2D (C, T). Got {data.shape}.")
        data = data.T
        for idx in range(data.shape[0]):
            name = f"{path.stem}:ch_{idx:03d}"
            yield ChannelSeries(name=name, values=data[idx], source=path)


def _extract_mat_matrix(mat: "h5py.File") -> np.ndarray:
    for key in mat.keys():
        dataset = mat[key]
        if isinstance(dataset, h5py.Dataset) and dataset.ndim == 2:
            data = np.array(dataset, dtype=float)
            if np.issubdtype(data.dtype, np.number):
                return data
    raise ValueError("No valid 2D numeric matrix found in .mat file.")
