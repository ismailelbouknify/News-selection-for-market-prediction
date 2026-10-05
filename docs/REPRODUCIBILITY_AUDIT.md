# Reproducibility audit

This document records how the released code relates to (a) the research
scripts that produced the GreenFin paper results and (b) the methodology as
described in the manuscript. Items marked **[discrepancy]** are places where
the code that produced the results and the manuscript text disagree; they are
reported here rather than silently "fixed" in the code, because changing them
would change the scientific results.

## 1. Provenance of the released code

The research experiments were run as self-contained scripts, each containing
an identical copy of the library code followed by a short block that set the
experiment configuration. The package in `src/greenfin/` is a refactor of that
library code. Function-by-function AST comparison against the final research
library shows that `Config`, the model (`GreenFin` = research `FININModel`),
dataset, collation, the four selectors, `preselect_news`, standardisation,
`train_loop`, `evaluate`, `get_returns_map` and the sliding-window CV are
**computationally identical**; differences are renamings, type hints, and
unused diagnostic lists in the research `train_loop`.

Changes made for the release (none alters a reported metric):

| Change | Reason |
|---|---|
| One checkpoint file per (seed, window) (`best_model_seed{S}_window{W}.pt`) | The research scripts passed one fixed checkpoint path to every window and seed, so each run overwrote the previous one. The in-memory metrics were unaffected, but no per-window checkpoints survive. If a window never selects a best epoch, the release uses the last-epoch weights instead of a stale file. |
| `kfold_time_cv` also returns per-day test predictions and per-window results | Needed for the bootstrap and practical trading analyses (the research scripts only printed aggregates). |
| Energy/CO2 totals are NaN, not 0, when CodeCarbon is disabled | Avoids reporting a fake zero. |
| The FININ "paper" metric is renamed `*_legacy_finin` | It is not a tradable strategy (see section 6). |
| `build_dataset.py` drops the last market day | That day has no next close; the research script kept it with label 0. It never enters any CV window (the T=5 windows end at sample 4,130 of 4,136), so results are unchanged. |
| `build_cleaned_news.py --keep-timestamp`, `build_dataset.py --alignment market_close` | Opt-in implementation of the market-close alignment described in the manuscript (section 3). Defaults reproduce the released data byte-for-byte (verified on synthetic input against the research builder). |
| Practical trading and bootstrap code moved into `greenfin.trading`, `greenfin.baselines`, `greenfin.bootstrap` with config-driven scripts | Previously separate analysis folders with hard-coded cluster paths. |

### Mapping from paper tables to configs

| Paper | Configs | Research scripts |
|---|---|---|
| Table 3, FININ-Full | `architecture/finin_full.yaml` | `Table0/T0_0..T0_4.py` (`use_miq=True`, all news, one seed per script) |
| Table 3, FININ-Selected | `architecture/finin_selected_farthest_k{10,50,100}.yaml` | `Table0/T11..T13.py` |
| Table 3, GreenFin-Full | `architecture/greenfin_full.yaml` | `Table2/T0_0..T0_4.py` (`use_miq=False`, all news) |
| Table 3/4, GreenFin-Selected | `selectors/farthest_k10.yaml` | `Table2/T11.py` |
| Table 3, MarketOnly | `architecture/greenfin_market_only.yaml` | `Table2/T1.py` |
| Table 4 | `selectors/{random,topconf,kmeans,farthest}_k{10,50,100}.yaml` | `Table2/T2..T13.py` |
| Table 6 | `lookback/farthest_T{1,3,5,10,20}_k{10,50,100}.yaml` | `Table3/T1..T15.py` |
| Table 7 | `configs/evaluation/bootstrap.yaml` | `statistical_validation/` |
| Table 8 | `configs/evaluation/trading.yaml` | `table_7/` |

`configs/experiments/legacy/finin_protocol_t5.yaml` corresponds to the
`Table1_finin` scripts (original FININ backtest protocol). It is not used in
the paper. Several of those research scripts used `cap_per_day=800,
news_select="random"` instead of all news; the release config uses all news.

## 2. Training configuration **[discrepancy]**

Every research script that produced Tables 3–6 used:

| Setting | Code (all 40 canonical scripts) | Manuscript |
|---|---|---|
| Optimiser | Adam | Adam |
| Learning rate | **1e-3** | 2e-4 |
| Batch size | **4** | 16 |
| Gradient accumulation | 16 (effective 64 samples per step) | 16 |
| Epochs | 50 (early stopping patience 100, i.e. inactive) | — |
| Loss | `BCEWithLogitsLoss(pos_weight = (1-p)/p)`, `p` = share of up days in the training split | BCEWithLogitsLoss |
| Gradient clipping | max-norm 0.5 | — |
| Model selection | epoch with best validation PnL (`best_select=val_pnl`) | — |

2e-4 and 16 are the library *defaults* of `Config`. The experiment scripts
overrode them, and `configs/base.yaml` uses the values that were run.

Architecture (from code): MLP encoders with 2 layers, hidden width 64, dropout
0.05 and Xavier initialisation (gain 0.5). Market encoder 5→32, sentiment
encoder 3→16, headline encoder 768→32, fusion →96 (3 heads × 32), each
followed by LayerNorm. GreenFin pools the fused headline vectors of a day with
a padding-masked mean; FININ-style models apply self-attention over the day's
headlines, then cross-attention with a market-derived query. The T daily
vectors (32 + 96 = 128 dims) are flattened and passed to an MLP head.
Multi-GPU execution uses `torch.nn.DataParallel` when more than one GPU is
visible.

## 3. Market/news time alignment **[discrepancy]**

**Code that produced the released data and all results:**
`build_cleaned_news.py` truncates each FNSPID timestamp to its **UTC calendar
date** (`pd.to_datetime(...).dt.date`). `build_dataset.py` then attaches to
trading day `d` all headlines whose date equals `d`. Headlines dated on
weekends or market holidays match no trading day and are **dropped**, not
rolled forward. There is no 16:00 New York cut-off.

**Manuscript:** 16:00 New York close cut-off; after-close and
weekend/holiday headlines move to the next trading day.

**Look-ahead check.** The label of day `d` is `Close_{d+1} > Close_d`. Every
headline assigned to `d` under the calendar policy carries a UTC date of `d`,
so it was published before the close of `d+1`. The released data therefore
has no look-ahead relative to the label. It does, however, include headlines
published after the close of `d` (until 19:00/20:00 New York time, depending on daylight saving) in day `d`'s
information set.

Raw FNSPID statistics (`nasdaq_exteral_data.csv`, 15,549,299 rows): 44.5% of
timestamps are exactly `00:00:00 UTC` (no intraday time), and 6.0% fall on a
UTC weekend.

The release implements the manuscript's policy in `greenfin.alignment`
(tested in `tests/test_alignment.py`) and exposes it through
`build_cleaned_news.py --keep-timestamp` and
`build_dataset.py --alignment market_close`. No experiment in the paper used
it. The authors should either describe the calendar policy in the manuscript
or regenerate the data with `market_close` and rerun the experiments.

## 4. FinBERT sentiment vector order **[discrepancy]**

`build_sentiment.py` stores FinBERT logits in columns named by the model's
`id2label` (`ProsusAI/finbert`: 0 = positive, 1 = negative, 2 = neutral).
`build_dataset.py` applies a softmax and writes each headline's vector as
**`[positive, negative, neutral]`**. The released JSONL files use this order,
for example the last sample: `[0.9346, 0.0169, 0.0485]`.

The manuscript describes the order negative / neutral / positive. The order
does not affect any result: the sentiment encoder is a learned linear map, and
topconf's entropy is permutation-invariant. The data was **not** reordered.
README and docstrings document the actual order.

FinBERT input text is `Article_title + ". " + Lsa_summary` (max 512 tokens).
RoBERTa (`roberta-base`, frozen) encodes `Article_title` only (max 50 tokens,
first-token `<s>` state of the last layer, stored as fp16).

## 5. Data period and market features **[discrepancy: period]**

* News is filtered to 2007-01-01 .. 2024-01-01. S&P 500 data (`^GSPC`, Yahoo
  Finance, `auto_adjust=False`) covers **2007-07-23 .. 2023-12-29**, i.e.
  4,140 trading days.
* Samples per look-back file: T=1: 4,140; T=3: 4,138; T=5: 4,136
  (2007-07-27 .. 2023-12-29); T=10: 4,131; T=20: 4,121. These counts include
  the final day with the invalid label (see section 1).
* T=5 CV windows: `(0, 2770), (340, 3110), (680, 3450), (1020, 3790),
  (1360, 4130)`, each with 2,216 / 277 / 277 train / validation / test days.
  The test windows are 2017-06-21..2018-07-26, 2018-10-25..2019-12-02,
  2020-03-05..2021-04-09, 2021-07-12..2022-08-15 and 2022-11-14..2023-12-20.
* The same geometry gives 5 windows for T=1, 3 and 10 but only **4 windows
  for T=20** (a fifth window would need 4,130 samples). Windows are defined by
  sample index, so test dates shift slightly between look-backs.

The manuscript says "2007–2024". The aligned study period ends on 2023-12-29.

**Market features.** The model input is the raw daily `[Open, High, Low,
Close, Volume]` levels for the T days of the window. There are no returns, log
prices or technical indicators. Features are z-scored per CV window with
mean and standard deviation pooled over all days of the training split
(`MarketStandardizer`).

## 6. Financial metrics

Primary metric (all headline results; `greenfin.metrics`): position
`+1/-1` from the predicted class, daily return `position × (Close_{d+1}/Close_d − 1)`,
**summed** PnL per test window, and annualised Sharpe
`(mean − 0.02/252)/std · √252`. Both are averaged over windows, then seeds.
This is the corrected tradable definition.

Legacy FININ metric (`metrics.legacy_finin_daily_returns`, reported only by
the legacy protocol as `pnl_legacy_finin`/`sharpe_legacy_finin`):
`(+1 if correct else −1) × r_d`. A correct short on a falling day books a
loss under this metric, so it is not a tradable P&L. By default its Sharpe is
not annualised. It is never used as a primary metric.

Practical trading (Table 8; `greenfin.trading`) uses a different, explicitly
documented convention. Returns are compounded. Sharpe is computed on daily
excess returns with rf = 2%/yr. Transaction costs are
`bps/10,000 × |Δposition|`, starting from cash, so a long↔short flip costs 2
units. Each metric is computed per (window, seed) and then averaged. The AR
lag is selected on validation Sharpe only. Buy-and-hold and Always-Up take
identical positions; Always-Up additionally has a directional accuracy.

## 7. Results that cannot be verified from archived artifacts

* The research scripts printed metrics to stdout, which was not saved.
  CodeCarbon ran with `save_to_file=False`. No CSV/JSON of the Table 3–6
  numbers (accuracy, PnL, Sharpe, training time, energy, CO2, latency) exists
  in the research workspace, so the manuscript values could not be
  re-derived. The release saves all of these to `outputs/runs/*/` and
  `results/`.
* **Statistical tests.** Because of the shared-checkpoint bug, only window 5
  checkpoints survived for FININ-Full, GreenFin-Full and MarketOnly. The
  bootstrap analysis found in the research workspace therefore compared
  GreenFin farthest k=10 (seed 0) against FININ-Full on **window 5 only**. It
  found **no significant difference**: ΔPnL = −0.069, 95% CI [−0.238, 0.093],
  p = 0.39; ΔSharpe = −0.45, p = 0.39; ΔDA = −0.012, p = 0.50. Farthest vs
  random k=10 (seed 0, 5 windows) was not significant either (PnL p = 0.10).
  This limited analysis does **not** support the manuscript statement that
  GreenFin's PnL and Sharpe improvements over FININ-Full are statistically
  significant. Because it covers one window and one GreenFin seed, it does not
  refute it either. Settling the question needs the full 5-window, 5-seed
  analysis from a rerun that saves predictions, which the release code does
  (`scripts/reproduce/table7_bootstrap.sh`).
* Hardware (GPU model/count) used for the paper runs is not recorded in any
  archived log.

## 8. Resource accounting

Downstream stage: CodeCarbon wraps the training loop of each (seed, window).
Inference latency is wall-clock time of test-set prediction, including batch
collation and embedding lookup, divided by the number of test days. It is
reported in ms per prediction, which equals seconds per 1,000 predictions.
Energy is not measured for inference.

Upstream stage: `build_sentiment.py` and `build_embeddings.py` measure the
FinBERT and RoBERTa inference loops and write
`results/carbon/upstream_*.json`.
`aggregate_results.py --upstream-carbon ...` adds the same upstream footprint
to every configuration for the conservative end-to-end accounting. The
efficiency reductions reported for GreenFin refer to the downstream stage
only.

## 9. Determinism

Seeds 0–4 set Python, NumPy and PyTorch RNGs. News selection for window `w`
uses seed `seed + w`. GPU kernels are not forced to be deterministic, so
reruns can differ slightly. The order of headlines within a day comes from
`pandas.sort_values("Date")` in `build_dataset.py`. It affects the random and
k-means selectors, so rebuild the data with the same code path to reproduce
the released files exactly.
