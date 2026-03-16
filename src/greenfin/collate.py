from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def collate_fn(
    batch: List[Dict[str, Any]],
    emb_dict: Dict[int, torch.Tensor],
    cap_per_day: Optional[int] = None,
    use_news: bool = True,
    selection_mode: str = "kmeans",
) -> Dict[str, Any]:
    """
    Collate function that only packs tensors.
    Assumes per-day headline capping/selection was already done by preselect_news(...).
    """
    if cap_per_day is not None:
        raise ValueError(
            "collate_fn must NOT cap when using preselect_news; pass cap_per_day=None here. "
            "Do all selection in preselect_news(ds, cap_per_day=..., selection_mode=...)."
        )

    batch_size = len(batch)
    t = batch[0]["markets"].shape[0]

    dates = [b["date"] for b in batch]
    markets = torch.stack([b["markets"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    if not use_news:
        return {"dates": dates, "markets": markets, "labels": labels}

    hmax = max((len(ids) for b in batch for ids in b["headline_ids"]), default=0)
    if hmax == 0:
        return {"dates": dates, "markets": markets, "labels": labels}

    try:
        e_in = int(next(iter(emb_dict.values())).shape[-1])
    except StopIteration:
        e_in = 1

    news_emb = torch.zeros((batch_size, t, hmax, e_in), dtype=torch.float32)
    sentiments = torch.zeros((batch_size, t, hmax, 3), dtype=torch.float32)
    pad_mask = torch.ones((batch_size, t, hmax), dtype=torch.bool)

    missing = 0
    for bi, b in enumerate(batch):
        for d in range(t):
            ids_d = b["headline_ids"][d]
            sents_d = b["sentiments"][d]
            use = min(len(ids_d), hmax)
            jj = 0
            for j in range(use):
                hid = int(ids_d[j])
                emb = emb_dict.get(hid)
                if emb is None:
                    missing += 1
                    continue
                news_emb[bi, d, jj] = emb.float().reshape(-1)
                sentiments[bi, d, jj] = torch.tensor(sents_d[j], dtype=torch.float32)
                pad_mask[bi, d, jj] = False
                jj += 1

    if missing:
        print(f"[collate] missing embeddings encountered: {missing}")

    return {
        "dates": dates,
        "markets": markets,
        "news_emb": news_emb,
        "sentiments": sentiments,
        "pad_mask": pad_mask,
        "labels": labels,
    }
