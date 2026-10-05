from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .windows import split_indices


class Sample:
    __slots__ = ("date", "markets", "headline_ids", "sentiments", "label")

    def __init__(
        self,
        date: str,
        markets: List[List[float]],
        headline_ids: Optional[List[List[int]]],
        sentiments: Optional[List[List[List[float]]]],
        label: int,
    ) -> None:
        self.date = date
        self.markets = markets
        self.headline_ids = headline_ids or []
        self.sentiments = sentiments or []
        self.label = label


def to_samples(records: Sequence[Dict[str, Any]]) -> List[Sample]:
    return [
        Sample(
            x["date"],
            x["markets"],
            x.get("headline_ids", []),
            x.get("sentiments", []),
            int(x["label"]),
        )
        for x in records
    ]


class GreenFinDataset(Dataset):
    def __init__(self, samples: Sequence[Sample | Dict[str, Any]]) -> None:
        if not samples:
            raise ValueError("GreenFinDataset requires non-empty samples.")
        if not isinstance(samples[0], Sample):
            samples = to_samples(samples)  # type: ignore[assignment]
        self.samples: List[Sample] = list(samples)  # type: ignore[arg-type]
        ts = {len(s.markets) for s in self.samples}
        if len(ts) != 1:
            raise ValueError(f"Mixed window sizes: {ts}")
        self.t = ts.pop()
        for s in self.samples:
            if s.headline_ids:
                assert len(s.headline_ids) == self.t, f"headline_ids len mismatch for {s.date}"
            if s.sentiments:
                assert len(s.sentiments) == self.t, f"sentiments len mismatch for {s.date}"
        self.samples.sort(key=lambda s: s.date)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "date": s.date,
            "markets": torch.tensor(s.markets, dtype=torch.float32),
            "headline_ids": s.headline_ids,
            "sentiments": s.sentiments,
            "label": torch.tensor([s.label], dtype=torch.float32),
        }


def make_splits(ds: GreenFinDataset, train: float = 0.8, val: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    """Chronological 80/10/10 split of a (date-sorted) window; see :mod:`greenfin.windows`."""
    return split_indices(len(ds), train=train, val=val)


def clone_dataset(ds: GreenFinDataset) -> GreenFinDataset:
    copies = []
    for s in ds.samples:
        copies.append(
            Sample(
                date=s.date,
                markets=[list(row) for row in s.markets],
                headline_ids=[list(day) for day in (s.headline_ids or [])],
                sentiments=[[list(x) for x in day] for day in (s.sentiments or [])],
                label=s.label,
            )
        )
    return GreenFinDataset(copies)
