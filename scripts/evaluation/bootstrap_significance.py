#!/usr/bin/env python
"""Paired block-bootstrap significance tests between models (paper Table 7).

For each comparison ``A vs B`` in the config, both models' ``predictions.csv``
files (written by ``scripts/experiments/run_experiment.py``) are matched on
(window, date). Each replicate resamples moving blocks (default length 5)
separately within every test window, applying the same draw to all seeds of
both models; see :mod:`greenfin.bootstrap` for the exact procedure.

Usage::

    python scripts/evaluation/bootstrap_significance.py --config configs/evaluation/bootstrap.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from greenfin.bootstrap import cells_from_predictions, paired_block_bootstrap


def _load(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"predictions file not found: {path}")
    return pd.read_csv(path, dtype={"date": str})


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/evaluation/bootstrap.yaml")
    p.add_argument("--n-boot", type=int, help="Override n_boot (e.g. a small value for a quick check).")
    args = p.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_boot = int(args.n_boot or cfg.get("n_boot", 10_000))
    rng_seed = int(cfg.get("rng_seed", 20260908))
    primary_block = int(cfg.get("block_length", 5))
    blocks = [primary_block] + [int(b) for b in cfg.get("sensitivity_block_lengths", []) if int(b) != primary_block]
    alpha = float(cfg.get("alpha", 0.05))
    out_dir = Path(cfg.get("output_dir", "results/significance"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for comp in cfg["comparisons"]:
        cells_a = cells_from_predictions(_load(comp["a"]))
        cells_b = cells_from_predictions(_load(comp["b"]))
        windows = comp.get("windows") or sorted({w for _, w in cells_a} & {w for _, w in cells_b})
        seeds_a = comp.get("seeds_a") or sorted({s for s, w in cells_a if w in windows})
        seeds_b = comp.get("seeds_b") or sorted({s for s, w in cells_b if w in windows})
        if not windows:
            raise ValueError(f"{comp['label']}: the two models share no test windows.")
        for block in blocks:
            res = paired_block_bootstrap(cells_a, cells_b, seeds_a, seeds_b, windows, block, n_boot, rng_seed)
            for r in res.values():
                rows.append({
                    "comparison": comp["label"],
                    "metric": r.metric,
                    "block_length": r.block_length,
                    "primary": block == primary_block,
                    "mean_a": r.observed_a,
                    "mean_b": r.observed_b,
                    "delta": r.observed_delta,
                    "ci95_low": r.ci_low,
                    "ci95_high": r.ci_high,
                    "p_value": r.p_value,
                    "significant": r.p_value < alpha,
                    "n_boot": r.n_boot,
                    "windows": " ".join(map(str, r.windows)),
                    "seeds_a": " ".join(map(str, r.seeds_a)),
                    "seeds_b": " ".join(map(str, r.seeds_b)),
                    "rng_seed": r.rng_seed,
                })
            print(f"[{comp['label']}] block={block}: " + ", ".join(
                f"{m} delta={r.observed_delta:+.4f} CI=[{r.ci_low:+.4f}, {r.ci_high:+.4f}] p={r.p_value:.4f}"
                for m, r in res.items()
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bootstrap_results.csv", index=False)
    print(f"\nWrote {out_dir}/bootstrap_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
