"""Every shipped experiment config must resolve to a valid Config matching the paper design."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs/experiments").rglob("*.yaml"))


def _runner():
    spec = importlib.util.spec_from_file_location("run_experiment", REPO / "scripts/experiments/run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(path: Path, base: str = "configs/base.yaml"):
    mod = _runner()
    args = mod.parse_args(["--base-config", str(REPO / base), "--experiment-config", str(path)])
    cfg = mod.resolve_config(args)
    return cfg, mod.build_runtime_config(cfg)


def test_expected_number_of_paper_configs():
    groups = {g: len(list((REPO / "configs/experiments" / g).glob("*.yaml"))) for g in ("architecture", "selectors", "lookback")}
    assert groups == {"architecture": 6, "selectors": 12, "lookback": 15}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(REPO / "configs/experiments")))
def test_config_resolves_and_matches_its_name(path):
    cfg, rc = _resolve(path)
    name = path.stem
    assert rc.use_miq == name.startswith("finin"), "only FININ configs may use the MIQ architecture"
    m = re.search(r"(random|topconf|kmeans|farthest)(?:_T(\d+))?_k(\d+)", name)
    if m:
        assert rc.news_select == m.group(1) and rc.cap_per_day == int(m.group(3))
        t = int(m.group(2) or 5)
    else:
        assert rc.cap_per_day is None and rc.news_select is None
        t = 5
    assert cfg["data"]["input_jsonl"].endswith(f"Input_t{t}.jsonl")
    assert rc.use_news == (name != "greenfin_market_only")
    names = {p.stem for p in CONFIGS}
    assert len(names) == len(CONFIGS)


def test_paper_training_settings_in_base_config():
    _, rc = _resolve(REPO / "configs/experiments/selectors/farthest_k10.yaml")
    assert (rc.epochs, rc.batch_size, rc.lr, rc.grad_accum_steps) == (50, 4, 1e-3, 16)
    assert rc.best_select == "val_pnl" and rc.rf_annual == 0.02 and rc.temporal == "mlp"
    cfg, _ = _resolve(REPO / "configs/experiments/selectors/farthest_k10.yaml")
    assert cfg["train"]["seed_list"] == [0, 1, 2, 3, 4]
    assert (cfg["cv"]["train_days"], cfg["cv"]["step_days"]) == (2430, 340)


def test_experiment_names_are_unique():
    mod = _runner()
    names = [mod.load_yaml(str(p))["experiment_name"] for p in CONFIGS]
    assert len(names) == len(set(names))
