from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pprint import pprint
from typing import Any, Dict

import torch
import yaml

from greenfin.config import Config
from greenfin.cv import FININevaluation, kfold_time_cv
from greenfin.dataset import GreenFinDataset


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_configs(base_path: str, experiment_path: str) -> Dict[str, Any]:
    base_cfg = load_yaml(base_path)
    exp_cfg = load_yaml(experiment_path)
    return deep_update(deepcopy(base_cfg), exp_cfg)


def build_runtime_config(cfg_dict: Dict[str, Any]) -> Config:
    train = cfg_dict.get("train", {})
    model = cfg_dict.get("model", {})
    task = cfg_dict.get("task", {})
    selection = cfg_dict.get("selection", {})
    checkpoint = cfg_dict.get("checkpoint", {})
    tracking = cfg_dict.get("tracking", {})

    return Config(
        # training
        epochs=train.get("epochs", 10),
        batch_size=train.get("batch_size", 16),
        lr=train.get("lr", 2e-4),
        amp=train.get("amp", False),
        seed=train.get("seed", 2),
        early_stop_patience=train.get("early_stop_patience", 100),

        # model
        v1=model.get("v1", 32),
        v2=model.get("v2", 16),
        v3=model.get("v3", 32),
        v4=model.get("v4", 32),
        mlp_layers=model.get("mlp_layers", 2),
        mlp_hidden=model.get("mlp_hidden", 64),
        attn_heads=model.get("attn_heads", 3),
        attn_head_dim=model.get("attn_head_dim", 32),
        temporal=model.get("temporal", "mlp"),

        # task
        decision_threshold=task.get("decision_threshold", 0.5),
        rf_annual=task.get("rf_annual", 0.02),
        use_market=task.get("use_market", True),
        use_news=task.get("use_news", True),
        use_sentiment=task.get("use_sentiment", True),
        use_miq=task.get("use_miq", True),
        loss_name=task.get("loss_name", "bce"),
        focal_alpha=task.get("focal_alpha", 1.0),
        focal_gamma=task.get("focal_gamma", 3.0),

        # selection / paths
        cap_per_day=selection.get("cap_per_day", None),
        news_select=selection.get("news_select", None),
        best_model_path=checkpoint.get("best_model_path", "outputs/checkpoints/default/best_model.pt"),
        best_select=checkpoint.get("best_select", "val_pnl"),
        use_best_for_test=checkpoint.get("use_best_for_test", True),
        carbon_log_dir=tracking.get("carbon_log_dir", "outputs/carbon"),
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GreenFin experiment from YAML configs.")
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/no_news.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged config before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_dict = merge_configs(args.base_config, args.experiment_config)

    if args.print_config:
        print("Merged config:")
        pprint(cfg_dict)
        print("-" * 80)

    experiment_name = cfg_dict.get("experiment_name", "experiment")
    data_cfg = cfg_dict.get("data", {})
    train_cfg = cfg_dict.get("train", {})
    cv_cfg = cfg_dict.get("cv", {})

    market_csv = data_cfg["market_csv"]
    input_jsonl = data_cfg["input_jsonl"]
    embeddings_path = data_cfg.get("embeddings_path")

    print(f"Starting experiment: {experiment_name}")

    data = load_jsonl(input_jsonl)
    ds = GreenFinDataset(data)

    runtime_cfg = build_runtime_config(cfg_dict)
    ensure_parent_dir(runtime_cfg.best_model_path)
    ensure_parent_dir(runtime_cfg.carbon_log_dir)

    if runtime_cfg.use_news:
        if not embeddings_path:
            raise ValueError("embeddings_path must be set when use_news=True")
        emb_dict = torch.load(embeddings_path, map_location="cpu")
    else:
        emb_dict = {}

    seed_list = train_cfg.get("seed_list", [runtime_cfg.seed])
    cv_method = cv_cfg.get("method", "kfold_time_cv")

    if cv_method == "FININevaluation":
        summary = FININevaluation(
            ds=ds,
            emb_dict=emb_dict,
            market_csv=market_csv,
            cfg=runtime_cfg,
            seed_list=seed_list,
            k=cv_cfg.get("k", 10),
            window_size=cv_cfg.get("window_size", 500),
            step=cv_cfg.get("step", 391),
            paper_annualise_sharpe=cv_cfg.get("paper_annualise_sharpe", False),
            paper_rf_is_annual=cv_cfg.get("paper_rf_is_annual", True),
            tdays=cv_cfg.get("tdays", 252),
        )
    elif cv_method == "kfold_time_cv":
        summary = kfold_time_cv(
            ds=ds,
            emb_dict=emb_dict,
            market_csv=market_csv,
            cfg=runtime_cfg,
            seed_list=seed_list,
            train_days=cv_cfg.get("train_days", 2430),
            step_days=cv_cfg.get("step_days", 340),
            max_windows=cv_cfg.get("max_windows", None),
        )
    else:
        raise ValueError(f"Unknown cv.method: {cv_method}")

    print(f"\nExperiment: {experiment_name}")
    pprint(summary)


if __name__ == "__main__":
    main()