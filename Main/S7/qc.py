from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class QCReport:
    candidate_count: int
    class_counts: Dict[int, int]
    p2p_stats: Dict[str, float]
    snr_stats: Dict[str, float]


def generate_qc_report(X: np.ndarray, pred: np.ndarray) -> QCReport:
    if X.ndim != 3:
        raise ValueError(f"X must have shape (C, N, t). Got {X.shape}.")
    if pred.ndim != 2:
        raise ValueError(f"pred must have shape (C, N). Got {pred.shape}.")
    if X.shape[:2] != pred.shape:
        raise ValueError("X and pred must align on (C, N).")

    valid_mask = ~np.isnan(X).any(axis=2)
    candidate_count = int(valid_mask.sum())

    classes = np.unique(pred[pred >= 0])
    class_counts: Dict[int, int] = {}
    for cls in classes:
        class_counts[int(cls)] = int((pred == cls).sum())

    p2p = _peak_to_peak(X)
    p2p_stats = _summary_stats(p2p)
    snr = _simple_snr(X)
    snr_stats = _summary_stats(snr)

    return QCReport(
        candidate_count=candidate_count,
        class_counts=class_counts,
        p2p_stats=p2p_stats,
        snr_stats=snr_stats,
    )


def _peak_to_peak(X: np.ndarray) -> np.ndarray:
    return np.nanmax(X, axis=2) - np.nanmin(X, axis=2)


def _simple_snr(X: np.ndarray) -> np.ndarray:
    mean = np.nanmean(X, axis=2)
    std = np.nanstd(X, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(mean) / std
    return snr


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p90": float(np.percentile(flat, 90)),
    }
