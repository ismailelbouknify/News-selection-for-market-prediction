from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


class MarketStandardizer:
    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, train_loader) -> None:
        m1 = m2 = None
        n = 0
        for b in train_loader:
            x = b["markets"]
            x = x.reshape(-1, x.shape[-1]).numpy()
            s1 = x.sum(axis=0)
            s2 = (x * x).sum(axis=0)
            if m1 is None:
                m1, m2 = s1, s2
            else:
                m1 += s1
                m2 += s2
            n += x.shape[0]
        mu = m1 / max(1, n)
        var = m2 / max(1, n) - mu * mu
        std = np.sqrt(np.maximum(var, 1e-8))
        std = np.maximum(std, 1e-4)
        self.mean = torch.tensor(mu, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


def to_device(x: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    return x.to(device, non_blocking=(device.type == "cuda")) if x is not None else None


def to_device_and_scale(
    batch: Dict[str, Any],
    device: torch.device,
    scaler: Optional[MarketStandardizer],
) -> Dict[str, Any]:
    markets = to_device(batch["markets"], device)
    if scaler is not None:
        markets = scaler.apply(markets)
    return {
        "dates": batch["dates"],
        "markets": markets,
        "news_emb": to_device(batch.get("news_emb"), device),
        "sentiments": to_device(batch.get("sentiments"), device),
        "pad_mask": to_device(batch.get("pad_mask"), device),
        "labels": to_device(batch["labels"], device),
    }
