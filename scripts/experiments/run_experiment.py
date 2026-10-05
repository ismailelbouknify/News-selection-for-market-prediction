#!/usr/bin/env python
"""Run one GreenFin / FININ experiment from YAML configuration.

Example (from the repository root)::

    python scripts/experiments/run_experiment.py \
        --experiment-config configs/experiments/selectors/farthest_k10.yaml

The experiment YAML is deep-merged on top of ``--base-config``. Outputs are
written to ``<output-root>/<experiment_name>/``:

* ``config.yaml``      -- the fully merged configuration that was run;
* ``summary.json``     -- per-seed and across-seed metrics, timings, energy, CO2;
* ``windows.csv``      -- one row per (seed, window) with the test metrics;
* ``predictions.csv``  -- one row per out-of-sample test day and (seed, window)
  (input to ``scripts/evaluation/bootstrap_significance.py`` and
  ``scripts/evaluation/practical_trading.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from greenfin.config import Config
from greenfin.cv import FININevaluation, kfold_time_cv
from greenfin.dataset import GreenFinDataset
from greenfin.io import configure_environment, load_embeddings, load_jsonl, sanity_check_embedding_coverage

ALLOWED_SECTIONS = {
    "experiment_name", "description", "data", "train", "model", "task",
    "selection", "checkpoint", "tracking", "cv", "output",
}

# YAML section/key -> Config field
CONFIG_KEYS = {
    "train": ["seed", "epochs", "batch_size", "lr", "grad_accum_steps", "amp", "early_stop_patience"],
    "model": ["temporal", "v1", "v2", "v3", "v4", "mlp_layers", "mlp_hidden", "attn_heads", "attn_head_dim"],
    "task": [
        "decision_threshold", "rf_annual", "use_market", "use_news", "use_sentiment", "use_miq",
        "loss_name", "focal_alpha", "focal_gamma",
    ],
    "selection": ["cap_per_day", "news_select"],
    "checkpoint": ["best_select", "use_best_for_test", "best_model_path"],
    "tracking": ["track_carbon", "carbon_save_to_file", "carbon_log_dir"],
}
EXTRA_KEYS = {
    "data": {"market_csv", "embeddings_path", "input_jsonl"},
    "train": {"seed_list"},
    "cv": {
        "method", "train_days", "step_days", "max_windows", "k", "window_size", "step",
        "legacy_annualise_sharpe", "legacy_rf_is_annual", "tdays",
    },
    "output": {"root"},
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return data or {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_keys(cfg: Dict[str, Any]) -> None:
    """Fail early on typos instead of silently running with defaults."""
    unknown_sections = set(cfg) - ALLOWED_SECTIONS
    if unknown_sections:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown_sections)}")
    for section, value in cfg.items():
        if not isinstance(value, dict):
            continue
        allowed = set(CONFIG_KEYS.get(section, [])) | EXTRA_KEYS.get(section, set())
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in section '{section}': {sorted(unknown)}")


def build_runtime_config(cfg: Dict[str, Any]) -> Config:
    kwargs: Dict[str, Any] = {"experiment_name": cfg.get("experiment_name", "experiment")}
    valid = {f.name for f in fields(Config)}
    for section, keys in CONFIG_KEYS.items():
        for key in keys:
            if key in cfg.get(section, {}):
                assert key in valid, key
                kwargs[key] = cfg[section][key]
    return Config(**kwargs)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", default="configs/base.yaml", help="Base YAML config.")
    p.add_argument("--experiment-config", required=True, help="Experiment YAML merged on top of the base config.")
    p.add_argument("--seeds", type=int, nargs="+", help="Override train.seed_list, e.g. --seeds 0 1 2 3 4.")
    p.add_argument("--output-root", help="Override output.root (default: outputs/runs).")
    p.add_argument("--data-dir", help="Prefix for the relative data paths in the config (default: repository root).")
    p.add_argument("--print-config", action="store_true", help="Print the merged config and exit.")
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deep_update(deepcopy(load_yaml(args.base_config)), load_yaml(args.experiment_config))
    validate_keys(cfg)
    if args.seeds:
        cfg.setdefault("train", {})["seed_list"] = list(args.seeds)
    if args.output_root:
        cfg.setdefault("output", {})["root"] = args.output_root
    if args.data_dir:
        for key in ("market_csv", "embeddings_path", "input_jsonl"):
            value = cfg.get("data", {}).get(key)
            if value and not Path(value).is_absolute():
                cfg["data"][key] = str(Path(args.data_dir) / value)
    return cfg


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(args)
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0

    configure_environment()
    name = cfg.get("experiment_name", "experiment")
    out_dir = Path(cfg.get("output", {}).get("root", "outputs/runs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    for key in ("market_csv", "input_jsonl"):
        if not data_cfg.get(key) or not Path(data_cfg[key]).exists():
            raise FileNotFoundError(f"data.{key} not found: {data_cfg.get(key)!r} (see README 'Data').")

    ckpt = cfg.setdefault("checkpoint", {})
    ckpt.setdefault("best_model_path", str(out_dir / "checkpoints" / "best_model.pt"))
    cfg.setdefault("tracking", {}).setdefault("carbon_log_dir", str(out_dir / "carbon"))
    runtime_cfg = build_runtime_config(cfg)
    print(f"Starting experiment: {name}  ->  {out_dir}")

    records = load_jsonl(data_cfg["input_jsonl"])
    ds = GreenFinDataset(records)

    if runtime_cfg.use_news:
        emb_path = data_cfg.get("embeddings_path")
        if not emb_path or not Path(emb_path).exists():
            raise FileNotFoundError(f"data.embeddings_path not found: {emb_path!r} (required when use_news=true).")
        # Keep fp16 storage (~22 GB for the full corpus); collate/selection cast to fp32 per batch.
        emb_dict = load_embeddings(emb_path, fp16_source=False)
        sanity_check_embedding_coverage(records, emb_dict)
    else:
        emb_dict = {}
    del records

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_cfg = cfg.get("train", {})
    cv_cfg = cfg.get("cv", {})
    seed_list = [int(s) for s in train_cfg.get("seed_list", [runtime_cfg.seed])]
    method = cv_cfg.get("method", "kfold_time_cv")
    predictions: List[Dict[str, Any]] = []

    if method == "kfold_time_cv":
        summary = kfold_time_cv(
            ds, emb_dict, data_cfg["market_csv"], runtime_cfg,
            seed_list=seed_list,
            train_days=int(cv_cfg.get("train_days", 2430)),
            step_days=int(cv_cfg.get("step_days", 340)),
            max_windows=cv_cfg.get("max_windows"),
            predictions_out=predictions,
        )
    elif method == "FININevaluation":
        summary = FININevaluation(
            ds, emb_dict, data_cfg["market_csv"], runtime_cfg,
            seed_list=seed_list,
            k=int(cv_cfg.get("k", 10)),
            window_size=int(cv_cfg.get("window_size", 500)),
            step=int(cv_cfg.get("step", 391)),
            legacy_annualise_sharpe=bool(cv_cfg.get("legacy_annualise_sharpe", False)),
            legacy_rf_is_annual=bool(cv_cfg.get("legacy_rf_is_annual", True)),
            tdays=int(cv_cfg.get("tdays", 252)),
        )
    else:
        raise ValueError(f"Unknown cv.method {method!r}; expected 'kfold_time_cv' or 'FININevaluation'.")

    per_seed = summary.get("per_seed_summaries", [summary])
    window_rows = [{"seed": s["outer_seed"], **w} for s in per_seed for w in s.get("windows", [])]
    pd.DataFrame(window_rows).to_csv(out_dir / "windows.csv", index=False)
    if predictions:
        cols = [
            "date", "seed", "window", "y_true", "prob_up", "predicted_class", "position",
            "next_day_return", "strategy_return", "selection_seed", "window_lo", "window_hi", "checkpoint",
        ]
        pd.DataFrame(predictions)[cols].to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"experiment_name": name, "cv_method": method, "summary": summary}, f, indent=2, default=float)

    print(f"\nExperiment {name} finished. Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
