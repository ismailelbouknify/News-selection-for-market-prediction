from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .collate import collate_fn
from .dataset import GreenFinDataset, make_splits
from .layers import FocalLoss
from .metrics import compute_pnl_sharpe
from .selection import preselect_news
from .standardize import MarketStandardizer, to_device_and_scale


class _Subset(Dataset):
    def __init__(self, base, idx):
        self.base = base
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def build_dataloaders(
    ds: GreenFinDataset,
    emb_dict: Dict[int, torch.Tensor],
    batch_size: int,
    cap: Optional[int] = None,
    use_news: bool = True,
    selection_mode: str = "kmeans",
):
    if use_news and cap is not None:
        preselect_news(ds, cap_per_day=cap, selection_mode=selection_mode, seed=42, emb_dict=emb_dict)

    if use_news:
        try:
            e_in = int(next(iter(emb_dict.values())).shape[-1])
        except StopIteration as exc:
            raise RuntimeError("Embedding dict is empty but use_news=True.") from exc
    else:
        e_in = 0

    idx_tr, idx_va, idx_te = make_splits(ds)
    tr = _Subset(ds, idx_tr)
    va = _Subset(ds, idx_va)
    te = _Subset(ds, idx_te)

    def _collate(b):
        return collate_fn(
            b,
            emb_dict,
            cap_per_day=None,
            use_news=use_news,
            selection_mode=selection_mode,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    F = _Subset(ds, [0])[0]["markets"].shape[-1]

    return (
        DataLoader(tr, batch_size, shuffle=True, collate_fn=_collate, num_workers=0, pin_memory=pin),
        DataLoader(va, batch_size, shuffle=False, collate_fn=_collate, num_workers=0, pin_memory=pin),
        DataLoader(te, batch_size, shuffle=False, collate_fn=_collate, num_workers=0, pin_memory=pin),
        e_in,
        F,
    )


def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int = 10,
    lr: float = 2e-4,
    amp: bool = False,
    threshold: float = 0.5,
    loss_name: str = "bce",
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    returns_map=None,
    rf_annual: float = 0.02,
    tdays: int = 252,
    grad_accum_steps: int = 8,
    scaler_ms: Optional[MarketStandardizer] = None,
    best_model_path: Optional[str] = "best_model.pt",
    pos_weight: Optional[float] = None,
    best_select: str = "val_loss",
    early_stop_patience: int = 20,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    computed_pos_weight = None
    if (loss_name == "bce") and (pos_weight is None):
        n_pos = n_tot = 0
        for b in train_loader:
            y = b["labels"].numpy()
            n_pos += int((y == 1).sum())
            n_tot += y.size
        p = max(1e-6, min(1 - 1e-6, n_pos / max(1, n_tot)))
        computed_pos_weight = float((1.0 - p) / p)

    if loss_name == "focal":
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        elif computed_pos_weight is not None:
            pw = torch.tensor([computed_pos_weight], dtype=torch.float32, device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_pnl": [], "val_pnl": [],
        "train_sharpe": [], "val_sharpe": [],
    }

    def _select_score(val_loss, val_acc, val_pnl, val_sharpe):
        sel = (best_select or "val_pnl").lower()
        if sel == "val_loss":
            return -float(val_loss)
        if sel == "val_acc":
            return float(val_acc)
        if sel == "val_sharpe":
            return float(val_sharpe) if (val_sharpe is not None and not np.isnan(val_sharpe)) else -float("inf")
        return float(val_pnl) if (val_pnl is not None and not np.isnan(val_pnl)) else -float("inf")

    best_score = -float("inf")
    best_epoch = 0

    def _maybe_save_best(ep_idx, val_loss, val_acc, val_pnl, val_sharpe):
        nonlocal best_score, best_epoch
        score = _select_score(val_loss, val_acc, val_pnl, val_sharpe)
        if score > best_score:
            best_score = score
            best_epoch = int(ep_idx)
            if best_model_path:
                to_save = model.module if isinstance(model, nn.DataParallel) else model
                os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)
                torch.save(to_save.state_dict(), best_model_path)
                print(f"[ckpt] Saved best ({best_select}) at epoch {ep_idx:02d} -> {best_model_path} | score={score:.6f}")

    best_val_loss_for_early = float("inf")
    epochs_without_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        if amp and device.type == "cuda":
            torch.cuda.empty_cache()

        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        micro_steps = 0
        n_correct_tr = 0
        n_total_tr = 0
        train_R: list[float] = []

        for i, raw_batch in enumerate(tqdm(train_loader, desc=f"Epoch {ep:02d} [train]", leave=False, dynamic_ncols=True)):
            batch = to_device_and_scale(raw_batch, device, scaler_ms)
            markets, news_emb = batch["markets"], batch.get("news_emb")
            sentiments, pad_mask, labels = batch.get("sentiments"), batch.get("pad_mask"), batch["labels"]

            ac = torch.amp.autocast("cuda") if (amp and device.type == "cuda") else nullcontext()
            with ac:
                logits, _ = model(markets, news_emb, sentiments, pad_mask)
                loss_full = loss_fn(logits.float(), labels.float())
                loss = loss_full / max(1, grad_accum_steps)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            run_loss += float(loss_full.detach().cpu())
            micro_steps += 1

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
                n_correct_tr += (preds == labels).sum().item()
                n_total_tr += labels.numel()

                if returns_map is not None:
                    s_vec = (preds * 2.0 - 1.0).view(-1).detach().cpu().numpy()
                    for i_b, d in enumerate(batch["dates"]):
                        r = returns_map.get(d)
                        if r is None or not np.isfinite(r):
                            continue
                        train_R.append(float(s_vec[i_b]) * float(r))

            do_step = (micro_steps % grad_accum_steps == 0) or (i == len(train_loader) - 1)
            if do_step:
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

        train_loss = run_loss / max(1, micro_steps)
        train_acc = n_correct_tr / max(1, n_total_tr)
        train_pnl, train_sharpe = compute_pnl_sharpe(train_R, rf_annual, tdays)

        model.eval()
        val_loss, vsteps = 0.0, 0
        n_correct = n_total = 0
        val_R: list[float] = []

        with torch.no_grad():
            for raw_batch in tqdm(val_loader, desc=f"Epoch {ep:02d} [val]  ", leave=False, dynamic_ncols=True):
                batch = to_device_and_scale(raw_batch, device, scaler_ms)
                markets, news_emb = batch["markets"], batch.get("news_emb")
                sentiments, pad_mask, labels = batch.get("sentiments"), batch.get("pad_mask"), batch["labels"]

                logits, _ = model(markets, news_emb, sentiments, pad_mask)
                l = loss_fn(logits.float(), labels.float())
                val_loss += float(l.detach().cpu())
                vsteps += 1

                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
                n_correct += (preds == labels).sum().item()
                n_total += labels.numel()

                if returns_map is not None:
                    s_vec = (preds * 2.0 - 1.0).view(-1).detach().cpu().numpy()
                    for i_b, d in enumerate(batch["dates"]):
                        r = returns_map.get(d)
                        if r is None or not np.isfinite(r):
                            continue
                        val_R.append(float(s_vec[i_b]) * float(r))

        val_loss /= max(1, vsteps)
        val_acc = n_correct / max(1, n_total)
        val_pnl, val_sharpe = compute_pnl_sharpe(val_R, rf_annual, tdays)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_pnl"].append(train_pnl)
        history["val_pnl"].append(val_pnl)
        history["train_sharpe"].append(train_sharpe)
        history["val_sharpe"].append(val_sharpe)

        print(
            f"Epoch {ep:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"train_pnl={train_pnl:.6f} | val_pnl={val_pnl:.6f} | "
            f"train_sharpe={np.nan if np.isnan(train_sharpe) else round(train_sharpe, 4)} | "
            f"val_sharpe={np.nan if np.isnan(val_sharpe) else round(val_sharpe, 4)}"
        )

        _maybe_save_best(ep, val_loss, val_acc, val_pnl, val_sharpe)

        if val_loss < best_val_loss_for_early - 1e-6:
            best_val_loss_for_early = val_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_stop_patience is not None and epochs_without_improve >= early_stop_patience:
            print(
                f"[early-stop] No improvement in val_loss for "
                f"{early_stop_patience} consecutive epochs. Stopping at epoch {ep:02d}."
            )
            break

    return history, float(best_score), int(best_epoch)
