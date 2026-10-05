"""Integration tests: standardisation, CV driver and the experiment CLI on tiny synthetic data."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from greenfin.config import Config
from greenfin.cv import kfold_time_cv, window_checkpoint_path
from greenfin.dataset import GreenFinDataset, make_splits
from greenfin.io import load_jsonl
from greenfin.standardize import MarketStandardizer
from greenfin.train import build_dataloaders

REPO = Path(__file__).resolve().parents[1]


def _load_script(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    _load_script("scripts/data/make_synthetic_data.py").main(
        ["--root", str(root), "--n-days", "130", "--lookbacks", "3", "--max-headlines", "8"]
    )
    return root


def test_market_standardizer_uses_training_split_only():
    rows = []
    for i in range(20):
        level = 1.0 if i < 16 else 1000.0  # validation/test days have extreme values
        rows.append({"date": f"2024-01-{i + 1:02d}", "markets": [[level] * 5], "headline_ids": [[]],
                     "sentiments": [[]], "label": i % 2})
    ds = GreenFinDataset(rows)
    train_idx, _, _ = make_splits(ds)
    assert len(train_idx) == 16
    train_loader, _, _, _, _ = build_dataloaders(ds, {}, batch_size=4, use_news=False)
    sc = MarketStandardizer()
    sc.fit(train_loader)
    np.testing.assert_allclose(sc.mean.numpy(), np.ones(5))


def test_window_checkpoint_paths_are_unique():
    paths = {window_checkpoint_path("out/ckpt/best_model.pt", s, w) for s in range(5) for w in range(1, 6)}
    assert len(paths) == 25
    assert window_checkpoint_path("out/best_model.pt", 3, 2) == str(Path("out/best_model_seed3_window2.pt"))


def test_kfold_time_cv_end_to_end(synthetic_root, tmp_path):
    ds = GreenFinDataset(load_jsonl(str(synthetic_root / "data/processed/Input_t3.jsonl")))
    emb = torch.load(synthetic_root / "data/interim/headline_embeddings_fp16.pt")
    cfg = Config(epochs=1, batch_size=4, grad_accum_steps=2, lr=1e-3, use_miq=False, cap_per_day=3,
                 news_select="farthest", track_carbon=False, best_model_path=str(tmp_path / "best.pt"))
    preds = []
    summary = kfold_time_cv(ds, emb, str(synthetic_root / "data/raw/market/sp500.csv"), cfg, seed_list=[0],
                            train_days=40, step_days=20, max_windows=2, predictions_out=preds)
    assert summary["k"] == 2
    assert len(preds) == 2 * 6  # 2 windows x 6 test days (60-sample windows)
    df = pd.DataFrame(preds)
    assert set(df["window"]) == {1, 2} and not df.duplicated(["window", "date"]).any()
    # PnL/Sharpe in the summary are recomputable from the saved daily predictions.
    for w in summary["windows"]:
        g = df[df["window"] == w["window"]]
        assert w["pnl"] == pytest.approx(g["strategy_return"].sum())
        assert w["acc"] == pytest.approx((g["predicted_class"] == g["y_true"]).mean())
    assert np.isnan(summary["energy_kWh_total"])  # tracking disabled -> NaN, never 0


def test_run_experiment_cli(synthetic_root, tmp_path):
    base = yaml.safe_load((REPO / "configs/smoke/base.yaml").read_text())
    base["train"].update({"seed_list": [0], "epochs": 1})
    base["cv"].update({"train_days": 40, "step_days": 20, "max_windows": 2})
    base["output"]["root"] = str(tmp_path / "runs")
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    exp = yaml.safe_load((REPO / "configs/experiments/architecture/finin_full.yaml").read_text())
    exp["data"]["input_jsonl"] = "data/processed/Input_t3.jsonl"
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp))

    mod = _load_script("scripts/experiments/run_experiment.py")
    assert mod.main(["--base-config", str(base_path), "--experiment-config", str(exp_path),
                     "--data-dir", str(synthetic_root)]) == 0
    out = tmp_path / "runs" / "finin_full"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cv_method"] == "kfold_time_cv"
    preds = pd.read_csv(out / "predictions.csv")
    assert {"date", "seed", "window", "y_true", "predicted_class", "strategy_return"} <= set(preds.columns)
    assert (out / "config.yaml").exists() and (out / "windows.csv").exists()


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        Config(news_select="best")
    with pytest.raises(ValueError):
        Config(cap_per_day=10, news_select=None)
    with pytest.raises(ValueError):
        Config(use_market=False, use_news=False)


def test_unknown_yaml_keys_are_rejected(tmp_path):
    mod = _load_script("scripts/experiments/run_experiment.py")
    with pytest.raises(ValueError, match="selection"):
        mod.validate_keys({"selection": {"cap_per_dya": 10}})


def test_predict_path_matches_original_evaluate(synthetic_root):
    from greenfin.evaluate import evaluate, predict, summarize_predictions
    from greenfin.metrics import get_returns_map
    from greenfin.model import GreenFin

    torch.manual_seed(0)
    ds = GreenFinDataset(load_jsonl(str(synthetic_root / "data/processed/Input_t3.jsonl")))
    emb = torch.load(synthetic_root / "data/interim/headline_embeddings_fp16.pt")
    _, _, test_loader, e_in, n_feats = build_dataloaders(ds, emb, batch_size=4, use_news=True)
    model = GreenFin(e_in=e_in, temporal="mlp", t=ds.t, use_miq=True, attn_heads=3, attn_head_dim=8,
                     n_market_feats=n_feats)
    sc = MarketStandardizer()
    sc.fit(test_loader)
    returns_map = get_returns_map(str(synthetic_root / "data/raw/market/sp500.csv"))
    device = torch.device("cpu")
    acc, pnl, sharpe, _ = evaluate(model, test_loader, device, returns_map, scaler_ms=sc)
    res = summarize_predictions(predict(model, test_loader, device, scaler_ms=sc), returns_map)
    assert res["acc"] == pytest.approx(acc)
    assert res["pnl"] == pytest.approx(pnl)
    assert res["sharpe"] == pytest.approx(sharpe)
