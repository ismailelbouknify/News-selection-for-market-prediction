# GreenFin: News Selection for Efficient Financial Market Prediction

Official repository for the project **"Towards Green AI in Finance: News Selection for Efficient Financial Market Prediction."**

This project studies whether a **small, selected subset of daily financial headlines** can preserve predictive signal while reducing **training time, inference latency, energy consumption, and CO2 emissions** in news-augmented financial market prediction.

---

## Overview

Financial forecasting pipelines often consume **all available daily news**, which can be expensive in terms of compute, energy, and carbon emissions. This repository introduces a **news selection stage** before downstream prediction and compares multiple selection strategies under a common evaluation pipeline.

The repository supports the following experiment settings:

- **No news**: market-only baseline
- **Full news**: all available headlines
- **TopConf**: confidence-based filtering using sentiment entropy
- **KMeans**: representative headline selection via clustering
- **Farthest**: diversity-based headline selection via farthest-point sampling

The full workflow includes:

1. News preprocessing and cleaning
2. S&P 500 market data download
3. FinBERT sentiment inference
4. RoBERTa headline embedding extraction
5. JSONL dataset creation for multiple lookback windows
6. Model training and time-series cross-validation
7. Trading and Green AI evaluation

---

## Repository Structure

```text
.
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── no_news.yaml
│       ├── full_news.yaml
│       ├── topconf.yaml
│       ├── kmeans.yaml
│       └── farthest.yaml
├── data/
│   ├── raw/
│   │   ├── market/
│   │   └── news/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── carbon/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── scripts/
│   ├── preprocess/
│   │   ├── build_cleaned_news.py
│   │   ├── download_sp500.py
│   │   └── preprocess_all.py
│   ├── download_data.py
│   ├── download_external_data.py
│   ├── build_sentiment.py
│   ├── build_embeddings.py
│   ├── build_dataset.py
│   └── run_experiment.py
├── src/
│   └── greenfin/
│       ├── __init__.py
│       ├── collate.py
│       ├── config.py
│       ├── cv.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── io.py
│       ├── layers.py
│       ├── metrics.py
│       ├── model.py
│       ├── selection.py
│       ├── standardize.py
│       └── train.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

### 1) Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Clone the Repository

First, clone the repository and move into the project folder:

```bash
git clone https://github.com/ismailelbouknify/News-selection-for-market-prediction.git
cd News-selection-for-market-prediction

## Datasets

### 1) Project dataset on Hugging Face

Processed experiment files and market data are hosted here:

**GreenFin dataset:**  
https://huggingface.co/datasets/ismail-ELBOUKNIFY/news-selection-for-market-prediction

This dataset currently provides:

- `processed/Input_t1.jsonl`
- `processed/Input_t3.jsonl`
- `processed/Input_t5.jsonl`
- `processed/Input_t10.jsonl`
- `processed/Input_t20.jsonl`
- `raw/market/sp500.csv`

### 2) External raw news dataset

The raw news file used for preprocessing is hosted separately here:

**FNSPID dataset:**  
https://huggingface.co/datasets/Zihan1004/FNSPID

This repository uses the external file:

- `data/raw/news/nasdaq_external_data.csv`

for full preprocessing and dataset regeneration.

---

## Data Setup

Data should be downloaded from Hugging Face.

Expected local structure after download:

```text
data/
├── raw/
│   ├── news/
│   │   └── nasdaq_external_data.csv
│   └── market/
│       └── sp500.csv
├── interim/
│   ├── cleaned_news_sentiment.csv
│   └── headline_embeddings_fp16.pt
└── processed/
    ├── Input_t1.jsonl
    ├── Input_t3.jsonl
    ├── Input_t5.jsonl
    ├── Input_t10.jsonl
    └── Input_t20.jsonl
```

### Option A: Download processed project data only

This is the fastest option if you want to run experiments directly.

```bash
python scripts/download_data.py
```

This downloads the processed JSONL datasets and market data from:

https://huggingface.co/datasets/ismail-ELBOUKNIFY/news-selection-for-market-prediction

### Option B: Download the external raw news file

If you want to reproduce the full preprocessing pipeline from raw news, run:

```bash
python scripts/download_external_data.py
```

This downloads the raw news file from:

https://huggingface.co/datasets/Zihan1004/FNSPID

### Recommended usage

- Use `download_data.py` if you want to **run experiments immediately**
- Use `download_external_data.py` if you want to **rebuild the pipeline from raw news**

---

## Quick Start

### Download processed data

```bash
python scripts/download_data.py
```

### Print experiment configuration

```bash
python scripts/run_experiment.py \
  --base-config configs/base.yaml \
  --experiment-config configs/experiments/no_news.yaml \
  --print-config
```

### Run the market-only baseline

```bash
python scripts/run_experiment.py \
  --experiment-config configs/experiments/no_news.yaml
```

---

## End-to-End Pipeline

There are two ways to use this repository.

### Path 1: Run experiments from downloaded processed data

If you only want to run the experiments, download the hosted project data first:

```bash
python scripts/download_data.py
```

Then run:

```bash
python scripts/run_experiment.py \
  --base-config configs/base.yaml \
  --experiment-config configs/experiments/no_news.yaml
```

Other experiment configs:

```bash
python scripts/run_experiment.py --experiment-config configs/experiments/full_news.yaml
python scripts/run_experiment.py --experiment-config configs/experiments/topconf.yaml
python scripts/run_experiment.py --experiment-config configs/experiments/kmeans.yaml
python scripts/run_experiment.py --experiment-config configs/experiments/farthest.yaml
```

### Path 2: Rebuild everything from raw data

If you want to fully regenerate the dataset from raw news:

#### Step 1: Download external raw news data

```bash
python scripts/download_external_data.py
```

#### Step 2: Clean raw news and download S&P 500 data

```bash
python scripts/preprocess/preprocess_all.py \
  --news-input data/raw/news/nasdaq_external_data.csv \
  --news-output data/interim/cleaned_news.csv \
  --market-output data/raw/market/sp500.csv \
  --ticker ^GSPC \
  --start 2007-07-23 \
  --end 2024-01-01 \
  --interval 1d
```

This step:

- cleans the raw news CSV
- removes duplicates and invalid dates
- removes rows containing Cyrillic text in the title/article fields
- downloads S&P 500 daily data from Yahoo Finance

#### Step 3: Run FinBERT sentiment inference

```bash
python scripts/build_sentiment.py
```

This script reads:

- `data/interim/cleaned_news.csv`

and writes:

- `data/interim/cleaned_news_sentiment.csv`

By default, this script uses:

- model: `ProsusAI/finbert`
- `REQUIRE_CUDA = True`

If you want CPU inference, edit `scripts/build_sentiment.py` and set:

```python
REQUIRE_CUDA = False
```

#### Step 4: Build headline embeddings with RoBERTa

```bash
python scripts/build_embeddings.py
```

This script reads:

- `data/interim/cleaned_news_sentiment.csv`

and writes:

- `data/interim/headline_embeddings_fp16.pt`

By default, this script uses:

- model: `roberta-base`
- text column: `Article_title`
- `REQUIRE_CUDA = True`

If needed, edit `scripts/build_embeddings.py` and set:

```python
REQUIRE_CUDA = False
```

#### Step 5: Build input JSONL datasets

```bash
python scripts/build_dataset.py
```

This script creates lookback datasets for:

- `t = 1`
- `t = 3`
- `t = 5`
- `t = 10`
- `t = 20`

and saves them under:

- `data/processed/Input_t1.jsonl`
- `data/processed/Input_t3.jsonl`
- `data/processed/Input_t5.jsonl`
- `data/processed/Input_t10.jsonl`
- `data/processed/Input_t20.jsonl`

#### Step 6: Run an experiment

```bash
python scripts/run_experiment.py \
  --base-config configs/base.yaml \
  --experiment-config configs/experiments/no_news.yaml
```

---

## Example Experiments

### Market-only baseline

```bash
python scripts/run_experiment.py \
  --experiment-config configs/experiments/no_news.yaml
```

### Full-news model

```bash
python scripts/run_experiment.py \
  --experiment-config configs/experiments/full_news.yaml
```

### TopConf headline selection

```bash
python scripts/run_experiment.py \
  --experiment-config configs/experiments/topconf.yaml
```

### KMeans headline selection

```bash
python scripts/run_experiment.py \
  --experiment-config configs/experiments/kmeans.yaml
```

### Farthest-point headline selection

```bash
python scripts/run_experiment.py \
  --experiment-config configs/experiments/farthest.yaml
```

---

## Configuration Overview

Main configuration sections are defined in `configs/base.yaml`:

- `data`: dataset and embedding paths
- `train`: seeds, epochs, batch size, learning rate, AMP, early stopping
- `model`: temporal backbone and hidden dimensions
- `task`: market/news usage, sentiment usage, MIQ, loss settings
- `selection`: daily news cap and selection strategy
- `checkpoint`: best-model selection metric and checkpoint path
- `tracking`: CodeCarbon output directory
- `cv`: time-series evaluation protocol

Supported temporal backbones:

- `mlp`
- `lstm`
- `cnn1d`

Supported selection modes:

- `topconf`
- `kmeans`
- `random`
- `farthest`

Supported CV modes:

- `kfold_time_cv`
- `FININevaluation`

---

## Dataset Format

Each line in the generated JSONL file has the following structure:

```json
{
  "date": "YYYY-MM-DD",
  "markets": [[...], [...], ...],
  "headline_ids": [[...], [...], ...],
  "sentiments": [[[positive, negative, neutral], ...], ...],
  "label": 0
}
```

Where:

- `date` is the target market day
- `markets` contains the rolling market window with features `[Open, High, Low, Close, Volume]`
- `headline_ids` stores per-day headline IDs aligned with the lookback window
- `sentiments` stores per-headline sentiment vectors aligned with `headline_ids`
- `label = 1` if `Close_{t+1} > Close_t`, otherwise `0`

---

## Outputs

Training and evaluation artifacts are written to:

```text
outputs/
├── carbon/         # CodeCarbon emissions logs
├── checkpoints/    # saved model checkpoints
├── logs/           # optional logs
└── results/        # experiment summaries / reports
```

The best checkpoint path is controlled by each experiment YAML file, for example:

```yaml
checkpoint:
  best_model_path: "outputs/checkpoints/topconf/best_model.pt"
```

---

## Running Tests

```bash
pytest -q
```

If you want to download the hosted project data before running experiments, use:

```bash
python scripts/download_data.py
```

If you want to reproduce preprocessing from raw news, use:

```bash
python scripts/download_external_data.py
```

---

## Notes

- Large files are hosted on Hugging Face instead of GitHub.
- The local `data/.cache/` folder created by Hugging Face downloads should be ignored by Git.
- If you use the external FNSPID news dataset, please follow its original license and citation requirements.

---

## Suggested Citation

If you use this repository, please cite the associated paper or project report. You can replace the placeholder entry below with the final bibliographic record:

```bibtex
@misc{greenfin_news_selection,
  title  = {Towards Green AI in Finance: News Selection for Efficient Financial Market Prediction},
  author = {Ismail ELBOUKNIFY},
  year   = {2026},
  note   = {GitHub repository}
}
```

---
