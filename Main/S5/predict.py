from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def predict(
    model: "nn.Module",
    X: np.ndarray,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for prediction.")
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, t). Got {X.shape}.")

    model.eval()
    expects_4d = _model_expects_4d(model)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            if expects_4d:
                batch = batch[:, None, None, :]
            logits = model(batch)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def _model_expects_4d(model: "nn.Module") -> bool:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return True
    return False
