from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker

from .config import Config
from .dataset import GreenFinDataset, clone_dataset
from .evaluate import evaluate, evaluate_always_buy
from .io import set_seed
from .metrics import get_returns_map
from .model import GreenFin, wrap_model_for_multi_gpu
from .selection import preselect_news
from .standardize import MarketStandardizer, to_device_and_scale
from .train import build_dataloaders, train_loop


def _mean_std_nan(xs: List[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


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
    paper_annualise_sharpe: bool = False,
    paper_rf_is_annual: bool = True,
    tdays: int = 252,
) -> Dict[str, Any]:
    if seed_list is None:
        seed_list = [getattr(cfg, "seed", 0)]
    orig_seed = getattr(cfg, "seed", 0)

    print(f"\n[FININevaluation] Running with seeds: {seed_list}\n")

    all_samples = sorted(ds.samples, key=lambda s: s.date)
    n = len(all_samples)
    assert n >= window_size, f"Need at least window_size={window_size}, but got n={n}."

    fold_bounds = []
    start_idx = 0
    while len(fold_bounds) < k and (start_idx + window_size) <= n:
        fold_bounds.append((start_idx, start_idx + window_size))
        start_idx += step
    num_windows = len(fold_bounds)

    print(f"[TimeCV] Sliding-window CV on {n} samples (window_size={window_size}, step={step}, windows={num_windows}).")
    print("Windows (lo, hi):", fold_bounds)

    returns_map = get_returns_map(market_csv)

    def _to_float_or_nan(x):
        try:
            if x is None:
                return float("nan")
            x = float(x)
            return x if np.isfinite(x) else float("nan")
        except Exception:
            return float("nan")

    @torch.no_grad()
    def _collect_daily_R(*, dates: List[str], preds: np.ndarray, labels: np.ndarray, mode: str) -> List[float]:
        R = []
        for i, d in enumerate(dates):
            r = returns_map.get(d)
            if r is None or not np.isfinite(r):
                continue
            if mode == "paper":
                flag = 1.0 if preds[i] == labels[i] else -1.0
                R.append(flag * float(r))
            else:
                s = 1.0 if preds[i] >= 1.0 else -1.0
                R.append(s * float(r))
        return R

    def _sharpe_from_R(R: List[float], *, rf_annual: float, annualise: bool, rf_is_annual: bool) -> float:
        if len(R) <= 2:
            return float("nan")
        arr = np.asarray(R, dtype=np.float64)
        vol = float(arr.std(ddof=1))
        if vol <= 0:
            return float("nan")
        mean = float(arr.mean())
        rf_daily = (rf_annual / tdays) if rf_is_annual else float(rf_annual)
        sr_daily = (mean - rf_daily) / vol
        return float(sr_daily * np.sqrt(tdays)) if annualise else float(sr_daily)

    @torch.no_grad()
    def evaluate_dual(model, loader, device, *, threshold: float, rf_annual: float, scaler_ms: Optional[MarketStandardizer]):
        model.eval()
        n_correct = 0
        n_total = 0
        R_strategy: List[float] = []
        R_paper: List[float] = []

        for raw_batch in loader:
            batch = to_device_and_scale(raw_batch, device, scaler_ms)
            logits, _ = model(batch["markets"], batch.get("news_emb"), batch.get("sentiments"), batch.get("pad_mask"))
            probs = torch.sigmoid(logits)
            preds_t = (probs >= threshold).float()
            labels_t = batch["labels"]

            n_correct += (preds_t == labels_t).sum().item()
            n_total += labels_t.numel()

            preds = preds_t.view(-1).detach().cpu().numpy()
            labels = labels_t.view(-1).detach().cpu().numpy()
            dates = batch["dates"]
            R_strategy.extend(_collect_daily_R(dates=dates, preds=preds, labels=labels, mode="strategy"))
            R_paper.extend(_collect_daily_R(dates=dates, preds=preds, labels=labels, mode="paper"))

        acc = (n_correct / n_total) if n_total else 0.0
        pnl_strategy = float(np.sum(R_strategy)) if len(R_strategy) else float("nan")
        pnl_paper = float(np.sum(R_paper)) if len(R_paper) else float("nan")
        sharpe_strategy = _sharpe_from_R(R_strategy, rf_annual=rf_annual, annualise=True, rf_is_annual=True)
        sharpe_paper = _sharpe_from_R(
            R_paper,
            rf_annual=rf_annual,
            annualise=paper_annualise_sharpe,
            rf_is_annual=paper_rf_is_annual,
        )
        return {
            "acc": float(acc),
            "pnl_strategy": float(pnl_strategy),
            "sharpe_strategy": float(sharpe_strategy),
            "pnl_paper": float(pnl_paper),
            "sharpe_paper": float(sharpe_paper),
        }

    def _run_single_seed(seed: int) -> Dict[str, Any]:
        print("=" * 80)
        print(f"[FININevaluation] Seed {seed}")
        cfg.seed = seed

        cuda_ok = set_seed(seed)
        device = torch.device("cuda" if (cuda_ok and torch.cuda.is_available()) else "cpu")
        if device.type == "cpu" and torch.cuda.is_available():
            print("[TimeCV] Proceeding on CPU because CUDA context looked unhealthy.")

        fold_results = []
        ab_metrics = []

        for fi, (lo, hi) in enumerate(fold_bounds, start=1):
            print("=" * 72)
            print(f"[Window {fi}/{num_windows}] samples [{lo}:{hi})")

            fold_ds = clone_dataset(GreenFinDataset(all_samples[lo:hi]))

            if cfg.use_news and cfg.cap_per_day is not None:
                preselect_news(
                    fold_ds,
                    cap_per_day=cfg.cap_per_day,
                    selection_mode=(cfg.news_select or "kmeans"),
                    seed=seed + fi,
                    emb_dict=emb_dict,
                )

            train_loader, val_loader, test_loader, e_in, F = build_dataloaders(
                fold_ds,
                emb_dict,
                cfg.batch_size,
                cap=None,
                use_news=cfg.use_news,
                selection_mode=(cfg.news_select or "kmeans"),
            )

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
                n_market_feats=F,
            ).to(device)
            model = wrap_model_for_multi_gpu(base_model)

            tracker = EmissionsTracker(save_to_file=False, log_level="error")
            tracker.start()
            t_train0 = time.time()

            _history, _best_score, best_epoch = train_loop(
                model, train_loader, val_loader, device,
                epochs=cfg.epochs, lr=cfg.lr, amp=cfg.amp,
                loss_name=cfg.loss_name, focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
                returns_map=returns_map, rf_annual=cfg.rf_annual,
                threshold=cfg.decision_threshold,
                grad_accum_steps=16,
                scaler_ms=scaler_ms,
                best_model_path=cfg.best_model_path,
                best_select=cfg.best_select,
                early_stop_patience=cfg.early_stop_patience,
            )

            train_time_min = (time.time() - t_train0) / 60.0
            energy_kWh = float("nan")
            co2_g = float("nan")
            try:
                co2_kg = float(tracker.stop())
                fed = getattr(tracker, "final_emissions_data", None)
                if fed is not None and hasattr(fed, "energy_consumed"):
                    energy_kWh = float(fed.energy_consumed)
                co2_g = co2_kg * 1000.0
            except Exception as e:
                print(f"[CodeCarbon] Warning in window {fi}: {e}")

            loaded_best = False
            if cfg.use_best_for_test and cfg.best_model_path and os.path.exists(cfg.best_model_path):
                try:
                    to_load = model.module if isinstance(model, nn.DataParallel) else model
                    state = torch.load(cfg.best_model_path, map_location=device)
                    to_load.load_state_dict(state, strict=True)
                    loaded_best = True
                    print(f"[ckpt] Loaded BEST checkpoint for window {fi} ({cfg.best_select} @epoch {best_epoch}) -> {cfg.best_model_path}")
                except Exception as e:
                    print(f"[ckpt] Warning: could not load best ({e}). Using last weights.")
            else:
                print(f"[ckpt] Using last weights for window {fi} (no best checkpoint found at {cfg.best_model_path}).")

            t_inf0 = time.time()
            dual = evaluate_dual(model, test_loader, device, threshold=cfg.decision_threshold, rf_annual=cfg.rf_annual, scaler_ms=scaler_ms)
            infer_time_s = time.time() - t_inf0
            n_test_days = len(test_loader.dataset)
            infer_time_per_day_ms = (infer_time_s * 1000.0) / max(1, n_test_days)

            ab_acc, ab_pnl, ab_sr = evaluate_always_buy(test_loader, returns_map, rf_annual=cfg.rf_annual)
            ab_metrics.append({"acc": float(ab_acc), "pnl": float(ab_pnl), "sharpe": float(ab_sr)})

            print(
                f"[Window {fi}] TEST (after {'BEST' if loaded_best else 'LAST'} weights) | "
                f"acc={dual['acc']:.4f}  pnl_strategy={dual['pnl_strategy']:.6f}  sharpe_strategy={dual['sharpe_strategy']:.4f}  "
                f"pnl_paper={dual['pnl_paper']:.6f}  sharpe_paper={dual['sharpe_paper']:.4f}  "
                f"train_time_min={train_time_min:.3f}  infer_time_s={infer_time_s:.3f}  "
                f"infer_time_per_day_ms={infer_time_per_day_ms:.4f}  energy_kWh={energy_kWh:.4f}  CO2_g={co2_g:.2f}"
            )
            print(f"[Window {fi}] AB (TEST): acc={ab_acc:.4f}  pnl={ab_pnl:.6f}  sharpe={ab_sr:.4f}")

            fold_results.append({
                "acc": float(dual["acc"]),
                "pnl_strategy": _to_float_or_nan(dual["pnl_strategy"]),
                "sharpe_strategy": _to_float_or_nan(dual["sharpe_strategy"]),
                "pnl_paper": _to_float_or_nan(dual["pnl_paper"]),
                "sharpe_paper": _to_float_or_nan(dual["sharpe_paper"]),
                "train_time_min": float(train_time_min),
                "infer_time_s": float(infer_time_s),
                "infer_time_per_day_ms": float(infer_time_per_day_ms),
                "energy_kWh": float(energy_kWh),
                "CO2_g": float(co2_g),
                "n_test_days": int(n_test_days),
            })

        accs = [r["acc"] for r in fold_results]
        pnlS = [r["pnl_strategy"] for r in fold_results]
        srS = [r["sharpe_strategy"] for r in fold_results]
        pnlP = [r["pnl_paper"] for r in fold_results]
        srP = [r["sharpe_paper"] for r in fold_results]

        acc_mean, _ = _mean_std_nan(accs)
        pnl_strategy_mean, _ = _mean_std_nan(pnlS)
        sharpe_strategy_mean, _ = _mean_std_nan(srS)
        pnl_paper_mean, _ = _mean_std_nan(pnlP)
        sharpe_paper_mean, _ = _mean_std_nan(srP)

        test_days_total = int(np.sum([r.get("n_test_days", 0) for r in fold_results]))
        train_time_min_total = float(np.nansum([r["train_time_min"] for r in fold_results]))
        infer_time_s_total = float(np.nansum([r["infer_time_s"] for r in fold_results]))
        infer_time_per_day_ms_total = float((infer_time_s_total * 1000.0) / max(1, test_days_total))
        energy_kWh_total = float(np.nansum([r["energy_kWh"] for r in fold_results]))
        CO2_g_total = float(np.nansum([r["CO2_g"] for r in fold_results]))

        ab_acc_mean = float(np.nanmean([m["acc"] for m in ab_metrics])) if ab_metrics else float("nan")
        ab_pnl_mean = float(np.nanmean([m["pnl"] for m in ab_metrics])) if ab_metrics else float("nan")
        ab_sr_mean = float(np.nanmean([m["sharpe"] for m in ab_metrics])) if ab_metrics else float("nan")

        return {
            "k": int(num_windows),
            "acc_mean": float(acc_mean),
            "pnl_strategy_mean": float(pnl_strategy_mean),
            "sharpe_strategy_mean": float(sharpe_strategy_mean),
            "pnl_paper_mean": float(pnl_paper_mean),
            "sharpe_paper_mean": float(sharpe_paper_mean),
            "test_days_total": float(test_days_total),
            "train_time_min_total": float(train_time_min_total),
            "infer_time_per_day_ms_total": float(infer_time_per_day_ms_total),
            "energy_kWh_total": float(energy_kWh_total),
            "CO2_g_total": float(CO2_g_total),
            "ab_acc_mean": float(ab_acc_mean),
            "ab_pnl_mean": float(ab_pnl_mean),
            "ab_sr_mean": float(ab_sr_mean),
        }

    per_seed_summaries: List[Dict[str, Any]] = []
    for seed in seed_list:
        s = _run_single_seed(seed)
        print("\n" + "=" * 80)
        print(f"[FININevaluation] Per-seed summary (seed={seed}, TEST aggregated over {int(s.get('k', 0))} windows):")
        for kk in [
            "acc_mean", "pnl_strategy_mean", "sharpe_strategy_mean", "pnl_paper_mean", "sharpe_paper_mean",
            "ab_acc_mean", "ab_pnl_mean", "ab_sr_mean", "train_time_min_total",
            "infer_time_per_day_ms_total", "energy_kWh_total", "CO2_g_total", "test_days_total",
        ]:
            print(f"  {kk}: {s.get(kk)}")
        print("=" * 80 + "\n")
        s2 = dict(s)
        s2["outer_seed"] = seed
        per_seed_summaries.append(s2)

    cfg.seed = orig_seed
    if len(seed_list) == 1:
        return per_seed_summaries[0]

    wanted = [
        "acc_mean", "pnl_strategy_mean", "sharpe_strategy_mean", "pnl_paper_mean", "sharpe_paper_mean",
        "ab_acc_mean", "ab_pnl_mean", "ab_sr_mean", "train_time_min_total", "infer_time_per_day_ms_total",
        "energy_kWh_total", "CO2_g_total", "test_days_total",
    ]
    out: Dict[str, Any] = {"seeds": seed_list, "per_seed_summaries": per_seed_summaries}
    for key in wanted:
        vals = np.asarray([s.get(key, np.nan) for s in per_seed_summaries], dtype=np.float64)
        out[f"{key}_avg"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))

    print("\n[FININevaluation] Across-seed aggregation (TEST):")
    for key in wanted:
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
) -> Dict[str, Any]:
    all_samples = sorted(ds.samples, key=lambda s: s.date)
    n = len(all_samples)

    window_size = train_days + step_days
    assert n >= window_size, f"Need at least {window_size} samples for one window, but got n={n}."

    fold_bounds = []
    start_idx = 0
    while (start_idx + window_size) <= n and (max_windows is None or len(fold_bounds) < max_windows):
        fold_bounds.append((start_idx, start_idx + window_size))
        start_idx += step_days

    num_windows = len(fold_bounds)
    print(f"[kfold_time_cv] Sliding-window CV on {n} samples (window_size={window_size}, step={step_days}, windows={num_windows}).")
    print("Windows (lo, hi):", fold_bounds)

    if seed_list is None:
        seed_list = [getattr(cfg, "seed", 0)]
    orig_seed = getattr(cfg, "seed", 0)
    returns_map = get_returns_map(market_csv)

    def _nanmean(x):
        return float(np.nanmean(np.asarray(x, dtype=np.float64))) if len(x) else float("nan")

    def _nanstd(x):
        return float(np.nanstd(np.asarray(x, dtype=np.float64))) if len(x) else float("nan")

    def _run_single_seed(seed: int) -> Dict[str, Any]:
        print("=" * 80)
        print(f"[kfold_time_cv] Seed {seed}")
        cfg.seed = seed

        cuda_ok = set_seed(seed)
        device = torch.device("cuda" if (cuda_ok and torch.cuda.is_available()) else "cpu")
        if device.type == "cpu" and torch.cuda.is_available():
            print("[kfold_time_cv] Proceeding on CPU because CUDA context looked unhealthy.")

        fold_results: List[Dict[str, Any]] = []
        ab_metrics: List[Dict[str, float]] = []

        for fi, (lo, hi) in enumerate(fold_bounds, start=1):
            print("=" * 72)
            print(f"[Window {fi}/{num_windows}] samples [{lo}:{hi})")

            fold_ds = clone_dataset(GreenFinDataset(all_samples[lo:hi]))

            if cfg.use_news and cfg.cap_per_day is not None:
                preselect_news(
                    fold_ds,
                    cap_per_day=cfg.cap_per_day,
                    selection_mode=(cfg.news_select or "kmeans"),
                    seed=seed + fi,
                    emb_dict=emb_dict,
                )

            train_loader, val_loader, test_loader, e_in, F = build_dataloaders(
                fold_ds,
                emb_dict,
                cfg.batch_size,
                cap=None,
                use_news=cfg.use_news,
                selection_mode=(cfg.news_select or "kmeans"),
            )

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
                n_market_feats=F,
            ).to(device)
            model = wrap_model_for_multi_gpu(base_model)

            tracker = EmissionsTracker(save_to_file=False, log_level="error")
            tracker.start()
            t_train0 = time.time()

            history, best_score, best_epoch = train_loop(
                model, train_loader, val_loader, device,
                epochs=cfg.epochs, lr=cfg.lr, amp=cfg.amp,
                loss_name=cfg.loss_name, focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
                returns_map=returns_map, rf_annual=cfg.rf_annual,
                threshold=cfg.decision_threshold,
                grad_accum_steps=16,
                scaler_ms=scaler_ms,
                best_model_path=cfg.best_model_path,
                best_select=cfg.best_select,
                early_stop_patience=cfg.early_stop_patience,
            )
            del history, best_score

            train_time_min = (time.time() - t_train0) / 60.0
            energy_kWh = float("nan")
            CO2_g = float("nan")
            try:
                co2_kg = float(tracker.stop())
                fed = getattr(tracker, "final_emissions_data", None)
                if fed is not None and hasattr(fed, "energy_consumed"):
                    energy_kWh = float(fed.energy_consumed)
                CO2_g = float(co2_kg * 1000.0)
            except Exception as e:
                print(f"[CodeCarbon] Warning in window {fi}: {e}")

            if cfg.use_best_for_test and cfg.best_model_path and os.path.exists(cfg.best_model_path):
                try:
                    to_load = model.module if isinstance(model, nn.DataParallel) else model
                    state = torch.load(cfg.best_model_path, map_location=device)
                    to_load.load_state_dict(state, strict=True)
                    print(f"[ckpt] Loaded best model ({cfg.best_select} @epoch {best_epoch}) from {cfg.best_model_path}")
                except Exception as e:
                    print(f"[ckpt] Warning: could not load best ({e}). Using last-epoch weights.")
            else:
                print(f"[ckpt] Using last-epoch weights for test (missing ckpt: {cfg.best_model_path}).")

            t_inf0 = time.time()
            acc, pnl, sr, _ = evaluate(
                model, test_loader, device, returns_map,
                threshold=cfg.decision_threshold, rf_annual=cfg.rf_annual, scaler_ms=scaler_ms,
            )
            infer_time_s = time.time() - t_inf0
            n_test_days = int(len(getattr(test_loader, "dataset", [])))
            infer_time_per_day_ms = (infer_time_s * 1000.0) / max(1, n_test_days)

            ab_acc, ab_pnl, ab_sr = evaluate_always_buy(test_loader, returns_map, rf_annual=cfg.rf_annual)
            ab_metrics.append({"acc": float(ab_acc), "pnl": float(ab_pnl), "sharpe": float(ab_sr)})

            fold_results.append({
                "window": fi,
                "acc": float(acc),
                "pnl": float(pnl) if pnl is not None else float("nan"),
                "sharpe": float(sr) if sr is not None else float("nan"),
                "train_time_min": float(train_time_min),
                "infer_time_s": float(infer_time_s),
                "infer_time_per_day_ms": float(infer_time_per_day_ms),
                "energy_kWh": float(energy_kWh),
                "CO2_g": float(CO2_g),
                "n_test_days": int(n_test_days),
            })

            print(
                f"[Window {fi}] acc={acc:.4f}  pnl={fold_results[-1]['pnl']:.6f}  sharpe={fold_results[-1]['sharpe']:.4f}  "
                f"train_time_min={train_time_min:.2f}  infer_time_s={infer_time_s:.2f}  "
                f"infer_time_per_day_ms={infer_time_per_day_ms:.4f}  energy_kWh={energy_kWh:.4f}  CO2_g={CO2_g:.2f}"
            )
            print(f"[Window {fi}] AB  : acc={ab_acc:.4f}  pnl={ab_pnl:.6f}  sharpe={ab_sr:.4f}")

        accs = [r["acc"] for r in fold_results]
        pnls = [r["pnl"] for r in fold_results]
        srs = [r["sharpe"] for r in fold_results]

        train_time_min_total = float(np.nansum([r["train_time_min"] for r in fold_results]))
        infer_time_s_total = float(np.nansum([r["infer_time_s"] for r in fold_results]))
        test_days_total = int(np.sum([r["n_test_days"] for r in fold_results]))
        infer_time_per_day_ms_total = float((infer_time_s_total * 1000.0) / max(1, test_days_total))
        energy_kWh_total = float(np.nansum([r["energy_kWh"] for r in fold_results]))
        CO2_g_total = float(np.nansum([r["CO2_g"] for r in fold_results]))

        ab_accs = [m["acc"] for m in ab_metrics]
        ab_pnls = [m["pnl"] for m in ab_metrics]
        ab_srs = [m["sharpe"] for m in ab_metrics]

        summary = {
            "k": int(num_windows),
            "acc_mean": _nanmean(accs), "acc_std": _nanstd(accs),
            "pnl_mean": _nanmean(pnls), "pnl_std": _nanstd(pnls),
            "sr_mean": _nanmean(srs), "sr_std": _nanstd(srs),
            "test_days_total": float(test_days_total),
            "train_time_min_total": float(train_time_min_total),
            "infer_time_per_day_ms_total": float(infer_time_per_day_ms_total),
            "energy_kWh_total": float(energy_kWh_total),
            "CO2_g_total": float(CO2_g_total),
            "ab_acc_mean": _nanmean(ab_accs), "ab_acc_std": _nanstd(ab_accs),
            "ab_pnl_mean": _nanmean(ab_pnls), "ab_pnl_std": _nanstd(ab_pnls),
            "ab_sr_mean": _nanmean(ab_srs), "ab_sr_std": _nanstd(ab_srs),
        }

        print(f"\n--- Sliding-window Summary (seed={seed}, {num_windows} windows) ---")
        print(summary)
        print(f"(Total energy: {energy_kWh_total:.4f} kWh, Total CO2: {CO2_g_total:.2f} g)\n")
        return summary

    per_seed_summaries: List[Dict[str, Any]] = []
    for s in seed_list:
        summ = _run_single_seed(int(s))
        summ2 = dict(summ)
        summ2["outer_seed"] = int(s)
        per_seed_summaries.append(summ2)

    cfg.seed = orig_seed
    if len(seed_list) == 1:
        return per_seed_summaries[0]

    wanted = [
        "acc_mean", "pnl_mean", "sr_mean", "ab_acc_mean", "ab_pnl_mean", "ab_sr_mean",
        "train_time_min_total", "infer_time_per_day_ms_total", "energy_kWh_total", "CO2_g_total", "test_days_total",
    ]
    out: Dict[str, Any] = {"seeds": seed_list, "per_seed_summaries": per_seed_summaries}
    for key in wanted:
        vals = np.asarray([ps.get(key, np.nan) for ps in per_seed_summaries], dtype=np.float64)
        out[f"{key}_avg"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))

    print("\n[kfold_time_cv] Aggregate metrics across seeds:")
    for key in wanted:
        print(f"  {key}: mean={out[key + '_avg']:.6f}, std={out[key + '_std']:.6f}")
    return out
