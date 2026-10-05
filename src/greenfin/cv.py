"""Time-series cross-validation drivers.

``kfold_time_cv`` implements the paper's evaluation protocol (see
:mod:`greenfin.windows`): 5 sliding windows of 2,770 trading days, step 340,
chronological 80/10/10 split inside each window, market-feature
standardisation fitted on each window's training split only, and daily news
selection applied per window with seed ``seed + window``. One model is trained
per (seed, window); the checkpoint with the best validation score
(``cfg.best_select``) is reloaded and evaluated on the window's test split.

``FININevaluation`` reproduces the original FININ backtest protocol (10
windows of 500 samples, stride 391). It is kept for comparison only and is not
used for any result in the GreenFin paper.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .dataset import GreenFinDataset, Sample, clone_dataset
from .evaluate import evaluate_always_buy, predict, summarize_predictions
from .io import set_seed
from .metrics import get_returns_map, legacy_finin_daily_returns
from .model import GreenFin, wrap_model_for_multi_gpu
from .selection import preselect_news
from .standardize import MarketStandardizer
from .train import build_dataloaders, train_loop
from .windows import build_fold_windows


def window_checkpoint_path(base_path: str, seed: int, window: int) -> str:
    """``outputs/checkpoints/x/best_model.pt`` -> ``.../best_model_seed{seed}_window{window}.pt``.

    Every (seed, window) gets its own file so that checkpoints are never
    overwritten by a later window or seed.
    """
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_seed{seed}_window{window}{p.suffix or '.pt'}"))


def _nanmean(x: List[float]) -> float:
    return float(np.nanmean(np.asarray(x, dtype=np.float64))) if len(x) else float("nan")


def _nanstd(x: List[float]) -> float:
    return float(np.nanstd(np.asarray(x, dtype=np.float64))) if len(x) else float("nan")


def _nansum(x: List[float]) -> float:
    """Sum ignoring NaN, but NaN (not 0) when nothing was measured (e.g. carbon tracking off)."""
    arr = np.asarray(x, dtype=np.float64)
    return float(np.nansum(arr)) if np.isfinite(arr).any() else float("nan")


def _start_tracker(cfg: Config, run_name: str):
    if not cfg.track_carbon:
        return None
    from codecarbon import EmissionsTracker

    kwargs: Dict[str, Any] = {"log_level": "error", "save_to_file": bool(cfg.carbon_save_to_file)}
    if cfg.carbon_save_to_file:
        Path(cfg.carbon_log_dir).mkdir(parents=True, exist_ok=True)
        kwargs.update(output_dir=cfg.carbon_log_dir, project_name=run_name)
    tracker = EmissionsTracker(**kwargs)
    tracker.start()
    return tracker


def _stop_tracker(tracker) -> Tuple[float, float]:
    """``(energy_kWh, CO2_g)`` of the tracked block; NaN when tracking is disabled."""
    if tracker is None:
        return float("nan"), float("nan")
    co2_kg = tracker.stop()
    data = getattr(tracker, "final_emissions_data", None)
    energy = float(data.energy_consumed) if data is not None and hasattr(data, "energy_consumed") else float("nan")
    co2_g = float(co2_kg) * 1000.0 if co2_kg is not None else float("nan")
    return energy, co2_g


def _fit_window(
    samples: List[Sample],
    emb_dict: Dict[int, torch.Tensor],
    cfg: Config,
    device: torch.device,
    returns_map: Dict[str, float],
    *,
    seed: int,
    window: int,
) -> Dict[str, Any]:
    """Select news, standardise, train and reload the best checkpoint for one (seed, window)."""
    fold_ds = clone_dataset(GreenFinDataset(samples))

    selection_seed: Optional[int] = None
    if cfg.use_news and cfg.cap_per_day is not None:
        # Selection is applied to each day's own candidate pool, so it never
        # uses information from other days (no cross-split leakage).
        selection_seed = seed + window
        preselect_news(
            fold_ds,
            cap_per_day=cfg.cap_per_day,
            selection_mode=(cfg.news_select or "kmeans"),
            seed=selection_seed,
            emb_dict=emb_dict,
        )

    train_loader, val_loader, test_loader, e_in, n_feats = build_dataloaders(
        fold_ds,
        emb_dict,
        cfg.batch_size,
        cap=None,
        use_news=cfg.use_news,
        selection_mode=(cfg.news_select or "kmeans"),
    )

    # Standardisation statistics come from the TRAINING split only.
    scaler_ms = MarketStandardizer()
    scaler_ms.fit(train_loader)

    base_model = GreenFin(
        e_in=max(e_in, 1),
        v1=cfg.v1, v2=cfg.v2, v3=cfg.v3, v4=cfg.v4,
        mlp_layers=cfg.mlp_layers, mlp_hidden=cfg.mlp_hidden,
        attn_heads=cfg.attn_heads, attn_head_dim=cfg.attn_head_dim,
        temporal=cfg.temporal, t=fold_ds.t,
        use_market=cfg.use_market, use_news=cfg.use_news,
        use_sentiment=cfg.use_sentiment, use_miq=cfg.use_miq,
        n_market_feats=n_feats,
    ).to(device)
    model = wrap_model_for_multi_gpu(base_model)

    ckpt_path = window_checkpoint_path(cfg.best_model_path, seed, window)
    tracker = _start_tracker(cfg, f"{cfg.experiment_name}_seed{seed}_window{window}")
    t_train0 = time.time()
    _history, _best_score, best_epoch = train_loop(
        model, train_loader, val_loader, device,
        epochs=cfg.epochs, lr=cfg.lr, amp=cfg.amp,
        loss_name=cfg.loss_name, focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
        returns_map=returns_map, rf_annual=cfg.rf_annual,
        threshold=cfg.decision_threshold,
        grad_accum_steps=cfg.grad_accum_steps,
        scaler_ms=scaler_ms,
        best_model_path=ckpt_path,
        best_select=cfg.best_select,
        early_stop_patience=cfg.early_stop_patience,
    )
    train_time_min = (time.time() - t_train0) / 60.0
    energy_kwh, co2_g = _stop_tracker(tracker)

    loaded_best = False
    if cfg.use_best_for_test and best_epoch > 0:
        to_load = model.module if isinstance(model, nn.DataParallel) else model
        to_load.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        loaded_best = True
        print(f"[ckpt] Loaded best ({cfg.best_select} @ epoch {best_epoch}) -> {ckpt_path}")
    else:
        print(f"[ckpt] Window {window}: using last-epoch weights (no best checkpoint was selected).")

    return {
        "model": model,
        "test_loader": test_loader,
        "scaler_ms": scaler_ms,
        "selection_seed": selection_seed,
        "checkpoint": ckpt_path if loaded_best else None,
        "best_epoch": int(best_epoch),
        "train_time_min": float(train_time_min),
        "energy_kWh": float(energy_kwh),
        "CO2_g": float(co2_g),
    }


def _seed_device(seed: int) -> torch.device:
    cuda_ok = set_seed(seed)
    device = torch.device("cuda" if (cuda_ok and torch.cuda.is_available()) else "cpu")
    if device.type == "cpu" and torch.cuda.is_available():
        print("[cv] Proceeding on CPU because the CUDA context looked unhealthy.")
    return device


def _aggregate_seeds(per_seed: List[Dict[str, Any]], keys: List[str], tag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seeds": [s["outer_seed"] for s in per_seed], "per_seed_summaries": per_seed}
    for key in keys:
        vals = np.asarray([s.get(key, np.nan) for s in per_seed], dtype=np.float64)
        out[f"{key}_avg"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
    print(f"\n[{tag}] Aggregate metrics across seeds:")
    for key in keys:
        print(f"  {key}: mean={out[key + '_avg']:.6f}, std={out[key + '_std']:.6f}")
    return out


def kfold_time_cv(
    ds: GreenFinDataset,
    emb_dict: Dict[int, torch.Tensor],
    market_csv: str,
    cfg: Config,
    seed_list: Optional[List[int]] = None,
    train_days: int = 2430,
    step_days: int = 340,
    max_windows: Optional[int] = None,
    predictions_out: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Paper protocol. Returns per-seed summaries (one seed) or their across-seed aggregate.

    If ``predictions_out`` is a list, one record per out-of-sample test day
    and (seed, window) is appended to it.
    """
    all_samples = sorted(ds.samples, key=lambda s: s.date)
    n = len(all_samples)
    fold_bounds = build_fold_windows(n, train_days, step_days, max_windows)
    num_windows = len(fold_bounds)
    print(
        f"[kfold_time_cv] Sliding-window CV on {n} samples "
        f"(window_size={train_days + step_days}, step={step_days}, windows={num_windows})."
    )
    print("Windows (lo, hi):", fold_bounds)

    if seed_list is None:
        seed_list = [cfg.seed]
    orig_seed = cfg.seed
    returns_map = get_returns_map(market_csv)

    def _run_single_seed(seed: int) -> Dict[str, Any]:
        print("=" * 80)
        print(f"[kfold_time_cv] Seed {seed}")
        cfg.seed = seed
        device = _seed_device(seed)

        fold_results: List[Dict[str, Any]] = []
        ab_metrics: List[Dict[str, float]] = []

        for fi, (lo, hi) in enumerate(fold_bounds, start=1):
            print("=" * 72)
            print(f"[Window {fi}/{num_windows}] samples [{lo}:{hi})")
            fit = _fit_window(all_samples[lo:hi], emb_dict, cfg, device, returns_map, seed=seed, window=fi)

            t_inf0 = time.time()
            records = predict(fit["model"], fit["test_loader"], device, cfg.decision_threshold, fit["scaler_ms"])
            res = summarize_predictions(records, returns_map, rf_annual=cfg.rf_annual)
            infer_time_s = time.time() - t_inf0
            n_test_days = len(records)
            infer_time_per_day_ms = (infer_time_s * 1000.0) / max(1, n_test_days)

            ab_acc, ab_pnl, ab_sr = evaluate_always_buy(fit["test_loader"], returns_map, rf_annual=cfg.rf_annual)
            ab_metrics.append({"acc": float(ab_acc), "pnl": float(ab_pnl), "sharpe": float(ab_sr)})

            fold_results.append({
                "window": fi,
                "test_start": records[0]["date"] if records else None,
                "test_end": records[-1]["date"] if records else None,
                "acc": res["acc"],
                "pnl": res["pnl"],
                "sharpe": res["sharpe"],
                "best_epoch": fit["best_epoch"],
                "train_time_min": fit["train_time_min"],
                "infer_time_s": float(infer_time_s),
                "infer_time_per_day_ms": float(infer_time_per_day_ms),
                "energy_kWh": fit["energy_kWh"],
                "CO2_g": fit["CO2_g"],
                "n_test_days": int(n_test_days),
            })
            if predictions_out is not None:
                for rec in records:
                    predictions_out.append({
                        **rec,
                        "seed": seed,
                        "window": fi,
                        "window_lo": lo,
                        "window_hi": hi,
                        "selection_seed": fit["selection_seed"],
                        "checkpoint": fit["checkpoint"],
                    })

            r = fold_results[-1]
            print(
                f"[Window {fi}] acc={r['acc']:.4f}  pnl={r['pnl']:.6f}  sharpe={r['sharpe']:.4f}  "
                f"train_time_min={r['train_time_min']:.2f}  infer_time_s={infer_time_s:.2f}  "
                f"infer_time_per_day_ms={infer_time_per_day_ms:.4f}  "
                f"energy_kWh={r['energy_kWh']:.4f}  CO2_g={r['CO2_g']:.2f}"
            )
            print(f"[Window {fi}] AB  : acc={ab_acc:.4f}  pnl={ab_pnl:.6f}  sharpe={ab_sr:.4f}")

        test_days_total = int(np.sum([r["n_test_days"] for r in fold_results]))
        infer_time_s_total = float(np.nansum([r["infer_time_s"] for r in fold_results]))
        summary = {
            "k": int(num_windows),
            "acc_mean": _nanmean([r["acc"] for r in fold_results]),
            "acc_std": _nanstd([r["acc"] for r in fold_results]),
            "pnl_mean": _nanmean([r["pnl"] for r in fold_results]),
            "pnl_std": _nanstd([r["pnl"] for r in fold_results]),
            "sr_mean": _nanmean([r["sharpe"] for r in fold_results]),
            "sr_std": _nanstd([r["sharpe"] for r in fold_results]),
            "test_days_total": float(test_days_total),
            "train_time_min_total": float(np.nansum([r["train_time_min"] for r in fold_results])),
            "infer_time_per_day_ms_total": float((infer_time_s_total * 1000.0) / max(1, test_days_total)),
            "energy_kWh_total": _nansum([r["energy_kWh"] for r in fold_results]),
            "CO2_g_total": _nansum([r["CO2_g"] for r in fold_results]),
            "ab_acc_mean": _nanmean([m["acc"] for m in ab_metrics]),
            "ab_acc_std": _nanstd([m["acc"] for m in ab_metrics]),
            "ab_pnl_mean": _nanmean([m["pnl"] for m in ab_metrics]),
            "ab_pnl_std": _nanstd([m["pnl"] for m in ab_metrics]),
            "ab_sr_mean": _nanmean([m["sharpe"] for m in ab_metrics]),
            "ab_sr_std": _nanstd([m["sharpe"] for m in ab_metrics]),
            "windows": fold_results,
        }
        print(f"\n--- Sliding-window summary (seed={seed}, {num_windows} windows) ---")
        print({k: v for k, v in summary.items() if k != "windows"})
        return summary

    per_seed_summaries: List[Dict[str, Any]] = []
    for s in seed_list:
        summ = _run_single_seed(int(s))
        summ["outer_seed"] = int(s)
        per_seed_summaries.append(summ)

    cfg.seed = orig_seed
    if len(seed_list) == 1:
        return per_seed_summaries[0]
    return _aggregate_seeds(
        per_seed_summaries,
        [
            "acc_mean", "pnl_mean", "sr_mean", "ab_acc_mean", "ab_pnl_mean", "ab_sr_mean",
            "train_time_min_total", "infer_time_per_day_ms_total", "energy_kWh_total", "CO2_g_total",
            "test_days_total",
        ],
        "kfold_time_cv",
    )


def FININevaluation(
    ds: GreenFinDataset,
    emb_dict: Dict[int, torch.Tensor],
    market_csv: str,
    cfg: Config,
    seed_list: Optional[List[int]] = None,
    k: int = 10,
    window_size: int = 500,
    step: int = 391,
    *,
    legacy_annualise_sharpe: bool = False,
    legacy_rf_is_annual: bool = True,
    tdays: int = 252,
) -> Dict[str, Any]:
    """Original FININ backtest protocol (legacy; NOT the GreenFin paper protocol).

    Reports the tradable long/short metric (``pnl_strategy``/``sharpe_strategy``,
    same definition as :mod:`greenfin.metrics`) and, for reproduction of the
    original FININ numbers only, the legacy metric
    (``pnl_legacy_finin``/``sharpe_legacy_finin``; by default a non-annualised
    Sharpe) -- see :func:`greenfin.metrics.legacy_finin_daily_returns`.
    """
    if seed_list is None:
        seed_list = [cfg.seed]
    orig_seed = cfg.seed

    all_samples = sorted(ds.samples, key=lambda s: s.date)
    n = len(all_samples)
    if n < window_size:
        raise ValueError(f"Need at least window_size={window_size} samples, got n={n}.")
    fold_bounds: List[Tuple[int, int]] = []
    start_idx = 0
    while len(fold_bounds) < k and (start_idx + window_size) <= n:
        fold_bounds.append((start_idx, start_idx + window_size))
        start_idx += step
    num_windows = len(fold_bounds)
    print(f"[FININevaluation] {n} samples, window_size={window_size}, step={step}, windows={num_windows}.")

    returns_map = get_returns_map(market_csv)

    def _legacy_sharpe(R: np.ndarray) -> float:
        if len(R) <= 2:
            return float("nan")
        vol = float(R.std(ddof=1))
        if vol <= 0:
            return float("nan")
        rf_daily = (cfg.rf_annual / tdays) if legacy_rf_is_annual else float(cfg.rf_annual)
        sr = (float(R.mean()) - rf_daily) / vol
        return float(sr * np.sqrt(tdays)) if legacy_annualise_sharpe else float(sr)

    def _run_single_seed(seed: int) -> Dict[str, Any]:
        cfg.seed = seed
        device = _seed_device(seed)
        fold_results: List[Dict[str, Any]] = []
        for fi, (lo, hi) in enumerate(fold_bounds, start=1):
            print(f"[FININevaluation] seed={seed} window {fi}/{num_windows} samples [{lo}:{hi})")
            fit = _fit_window(all_samples[lo:hi], emb_dict, cfg, device, returns_map, seed=seed, window=fi)
            t_inf0 = time.time()
            records = predict(fit["model"], fit["test_loader"], device, cfg.decision_threshold, fit["scaler_ms"])
            strategy = summarize_predictions(records, returns_map, rf_annual=cfg.rf_annual)
            infer_time_s = time.time() - t_inf0
            valid = [r for r in records if np.isfinite(r["next_day_return"])]
            legacy_R = legacy_finin_daily_returns(
                [r["predicted_class"] for r in valid], [r["y_true"] for r in valid], [r["next_day_return"] for r in valid]
            )
            fold_results.append({
                "window": fi,
                "acc": strategy["acc"],
                "pnl_strategy": strategy["pnl"],
                "sharpe_strategy": strategy["sharpe"],
                "pnl_legacy_finin": float(legacy_R.sum()) if legacy_R.size else float("nan"),
                "sharpe_legacy_finin": _legacy_sharpe(legacy_R),
                "train_time_min": fit["train_time_min"],
                "infer_time_s": float(infer_time_s),
                "energy_kWh": fit["energy_kWh"],
                "CO2_g": fit["CO2_g"],
                "n_test_days": len(records),
            })
        keys = ["acc", "pnl_strategy", "sharpe_strategy", "pnl_legacy_finin", "sharpe_legacy_finin"]
        summary: Dict[str, Any] = {f"{key}_mean": _nanmean([r[key] for r in fold_results]) for key in keys}
        summary.update({
            "k": int(num_windows),
            "test_days_total": float(np.sum([r["n_test_days"] for r in fold_results])),
            "train_time_min_total": float(np.nansum([r["train_time_min"] for r in fold_results])),
            "energy_kWh_total": _nansum([r["energy_kWh"] for r in fold_results]),
            "CO2_g_total": _nansum([r["CO2_g"] for r in fold_results]),
            "windows": fold_results,
        })
        return summary

    per_seed: List[Dict[str, Any]] = []
    for s in seed_list:
        summ = _run_single_seed(int(s))
        summ["outer_seed"] = int(s)
        per_seed.append(summ)
    cfg.seed = orig_seed
    if len(seed_list) == 1:
        return per_seed[0]
    return _aggregate_seeds(
        per_seed,
        [
            "acc_mean", "pnl_strategy_mean", "sharpe_strategy_mean", "pnl_legacy_finin_mean",
            "sharpe_legacy_finin_mean", "train_time_min_total", "energy_kWh_total", "CO2_g_total",
        ],
        "FININevaluation",
    )


__all__ = ["FININevaluation", "kfold_time_cv", "window_checkpoint_path"]
