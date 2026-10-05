from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .metrics import compute_pnl_sharpe, positions_from_predictions
from .standardize import MarketStandardizer, to_device_and_scale


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    returns_map,
    threshold: float = 0.5,
    rf_annual: float = 0.02,
    tdays: int = 252,
    collect_attn: bool = False,
    scaler_ms: Optional[MarketStandardizer] = None,
):
    model.eval()
    n_correct = 0
    n_total = 0
    trading_returns: list[float] = []
    all_attn = [] if collect_attn else None

    for raw_batch in loader:
        batch = to_device_and_scale(raw_batch, device, scaler_ms)
        markets = batch["markets"]
        news_emb = batch.get("news_emb")
        sentiments = batch.get("sentiments")
        pad_mask = batch.get("pad_mask")
        labels = batch["labels"]

        logits, attn_weights = model(markets, news_emb, sentiments, pad_mask)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        n_correct += (preds == labels).sum().item()
        n_total += labels.numel()

        s = (preds * 2.0 - 1.0).view(-1).cpu().numpy()
        for i, d in enumerate(batch["dates"]):
            r = returns_map.get(d)
            if r is None or not np.isfinite(r):
                continue
            trading_returns.append(float(s[i]) * float(r))

        if collect_attn:
            all_attn.append(attn_weights.detach().cpu() if attn_weights is not None else None)

    acc = n_correct / n_total if n_total else 0.0

    pnl = None
    sharpe = None
    if len(trading_returns) > 2:
        R = np.array(trading_returns, dtype=np.float64)
        pnl = float(R.sum())
        vol = float(R.std(ddof=1))
        mean = float(R.mean())
        if vol > 0:
            rf_daily = rf_annual / tdays
            sharpe_daily = (mean - rf_daily) / vol
            sharpe = sharpe_daily * np.sqrt(tdays)
        else:
            sharpe = float("nan")

    return acc, pnl, sharpe, all_attn


@torch.no_grad()
def evaluate_always_buy(val_or_test_loader, returns_map, rf_annual: float = 0.02, tdays: int = 252):
    n_up = 0
    n_total = 0
    R: list[float] = []

    for batch in val_or_test_loader:
        labels = batch["labels"]
        n_up += int((labels == 1).sum().item())
        n_total += labels.numel()

        for d in batch["dates"]:
            r = returns_map.get(d)
            if r is None or not np.isfinite(r):
                continue
            R.append(float(r))

    acc = (n_up / n_total) if n_total else 0.0
    pnl = float("nan")
    sharpe = float("nan")

    if len(R) > 2:
        R_arr = np.asarray(R, dtype=np.float64)
        pnl = float(R_arr.sum())
        vol = float(R_arr.std(ddof=1))
        mean = float(R_arr.mean())
        if vol > 0:
            rf_daily = rf_annual / tdays
            sharpe_daily = (mean - rf_daily) / vol
            sharpe = sharpe_daily * np.sqrt(tdays)
        else:
            sharpe = float("nan")

    return acc, pnl, sharpe


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    threshold: float = 0.5,
    scaler_ms: Optional[MarketStandardizer] = None,
) -> List[Dict[str, Any]]:
    """Per-sample predictions ``{date, y_true, prob_up, predicted_class}`` in loader order."""
    model.eval()
    records: List[Dict[str, Any]] = []
    for raw_batch in loader:
        batch = to_device_and_scale(raw_batch, device, scaler_ms)
        logits, _ = model(batch["markets"], batch.get("news_emb"), batch.get("sentiments"), batch.get("pad_mask"))
        probs = torch.sigmoid(logits).view(-1).cpu().numpy()
        labels = batch["labels"].view(-1).cpu().numpy()
        for d, y, p in zip(batch["dates"], labels, probs):
            records.append(
                {"date": d, "y_true": float(y), "prob_up": float(p), "predicted_class": float(p >= threshold)}
            )
    return records


def summarize_predictions(
    records: List[Dict[str, Any]],
    returns_map: Dict[str, float],
    rf_annual: float = 0.02,
    tdays: int = 252,
) -> Dict[str, float]:
    """Accuracy / PnL / Sharpe from :func:`predict` output (same definitions as :func:`evaluate`).

    Adds ``next_day_return``, ``position`` and ``strategy_return`` to every
    record in place (NaN where the next-day return is unavailable).
    """
    n_correct = 0
    trading_returns: List[float] = []
    for rec in records:
        n_correct += int(rec["predicted_class"] == rec["y_true"])
        position = float(positions_from_predictions([rec["predicted_class"]])[0])
        r = returns_map.get(rec["date"])
        rec["position"] = position
        if r is None or not np.isfinite(r):
            rec["next_day_return"] = float("nan")
            rec["strategy_return"] = float("nan")
            continue
        rec["next_day_return"] = float(r)
        rec["strategy_return"] = position * float(r)
        trading_returns.append(rec["strategy_return"])
    acc = n_correct / len(records) if records else 0.0
    pnl, sharpe = compute_pnl_sharpe(trading_returns, rf_annual, tdays)
    return {"acc": float(acc), "pnl": float(pnl), "sharpe": float(sharpe)}
