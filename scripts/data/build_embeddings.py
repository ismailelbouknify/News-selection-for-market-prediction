"""RoBERTa headline embeddings (upstream encoding stage).

Encodes ``Article_title`` with frozen ``roberta-base`` (first-token ``<s>``
representation of the last hidden layer, max 50 tokens) and saves
``{headline_id: float16 tensor[768]}``. ``headline_id`` is the 1-based row
number of the input CSV, the same id used by ``build_dataset.py``.
Energy/CO2 are measured with CodeCarbon and written to ``--carbon-json``.

Usage::

    python scripts/data/build_embeddings.py
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# =======================
# Config
# =======================
INPUT_CSV = "data/interim/cleaned_news_sentiment.csv"
OUTPUT_EMB = "data/interim/headline_embeddings_fp16.pt"
TEXT_COL = "Article_title"
MODEL_NAME = "roberta-base"
MAX_LEN = 50
BATCH_SIZE = 10240
REQUIRE_CUDA = True
# =======================


@dataclass(frozen=True)
class Config:
    input_csv: str = INPUT_CSV
    text_col: str = TEXT_COL
    output_emb: str = OUTPUT_EMB
    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    batch_size: int = BATCH_SIZE
    require_cuda: bool = REQUIRE_CUDA


def get_device(require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if require_cuda:
        raise RuntimeError(
            "CUDA GPU required for this script. "
            "Pass --allow-cpu to run on CPU."
        )
    return torch.device("cpu")


def load_input_dataframe(input_csv: str, text_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {input_csv}")

    # Keep row order exactly as-is so headline_id matches downstream scripts.
    df = df.copy().reset_index(drop=True)
    df["headline_id"] = range(1, len(df) + 1)
    return df


def build_texts_and_ids(df: pd.DataFrame, text_col: str) -> Tuple[List[str], List[int]]:
    texts = df[text_col].fillna("").astype(str).tolist()
    ids = df["headline_id"].astype(int).tolist()
    return texts, ids


def build_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[AutoTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    model = model.to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: nn.Module,
    device: torch.device,
    max_len: int,
) -> torch.Tensor:
    """
    Encode a batch of texts with RoBERTa and return the first-token representation.
    For RoBERTa, this corresponds to the <s> token position.
    """
    if not texts:
        return torch.empty((0, 768), dtype=torch.float32)

    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    tokens = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in tokens.items()}

    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state[:, 0, :]  # first token / <s>
    return embeddings.detach().cpu()


def warmup_model(model: nn.Module, tokenizer: AutoTokenizer, device: torch.device, max_len: int) -> None:
    if device.type != "cuda":
        return

    dummy_text = ["warmup"]
    with torch.no_grad():
        _ = encode_batch(dummy_text, tokenizer, model, device, max_len)
        torch.cuda.synchronize()


@torch.no_grad()
def run_encoding(
    texts: List[str],
    ids: List[int],
    tokenizer: AutoTokenizer,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    max_len: int,
) -> Tuple[Dict[int, torch.Tensor], float, float, float]:
    tracker = EmissionsTracker(save_to_file=False, log_level="error")
    emb_dict: Dict[int, torch.Tensor] = {}

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    tracker.start()

    try:
        for start in tqdm(range(0, len(texts), batch_size), desc="RoBERTa encoding"):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]

            batch_embs = encode_batch(
                texts=batch_texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_len=max_len,
            )

            for hid, emb in zip(batch_ids, batch_embs):
                emb_dict[int(hid)] = emb.half()
    finally:
        co2_kg = tracker.stop()
        if device.type == "cuda":
            torch.cuda.synchronize()

    encoding_time_s = time.perf_counter() - t0

    energy_kwh = float("nan")
    co2_g = float("nan")

    try:
        if co2_kg is not None and math.isfinite(float(co2_kg)):
            co2_g = float(co2_kg) * 1000.0

        emissions_data = getattr(tracker, "final_emissions_data", None)
        if emissions_data is not None and getattr(emissions_data, "energy_consumed", None) is not None:
            energy_kwh = float(emissions_data.energy_consumed)
    except Exception:
        pass

    return emb_dict, encoding_time_s, energy_kwh, co2_g


def save_embeddings(emb_dict: Dict[int, torch.Tensor], output_emb: str) -> None:
    out_dir = os.path.dirname(output_emb)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(emb_dict, output_emb)


def print_summary(
    n_rows: int,
    encoding_time_s: float,
    total_time_s: float,
    energy_kwh: float,
    co2_g: float,
    output_emb: str,
) -> None:
    print(f"\nSaved {n_rows:,} embeddings to {output_emb} (float16)")
    print("---- Timing ----")
    print(f"Encoding time: {encoding_time_s:.2f} s  ({encoding_time_s / 60.0:.2f} min)")
    print(f"Total time   : {total_time_s:.2f} s  ({total_time_s / 60.0:.2f} min)")
    print("---- CodeCarbon ----")
    print(f"Energy (kWh) : {energy_kwh:.6f}")
    print(f"CO2 (g)      : {co2_g:.2f}")
    print("---- Per headline ----")
    print(f"Rows         : {n_rows:,}")
    print(f"Time/headline (ms): {(encoding_time_s * 1000.0) / max(1, n_rows):.6f}")
    print(f"Energy/headline (Wh): {(energy_kwh * 1000.0) / max(1, n_rows):.10f}")
    print(f"CO2/headline (mg): {(co2_g * 1000.0) / max(1, n_rows):.10f}")


def parse_args() -> tuple[Config, str]:
    import argparse

    default = Config()
    parser = argparse.ArgumentParser(description=__doc__ or "embeddings")
    parser.add_argument("--input-csv", type=str, default=default.input_csv, help="News CSV (row order defines headline_id) (default: %(default)s)")
    parser.add_argument("--output-emb", type=str, default=default.output_emb, help="Output .pt file (default: %(default)s)")
    parser.add_argument("--text-col", type=str, default=default.text_col, help="Text column to encode (default: %(default)s)")
    parser.add_argument("--model-name", type=str, default=default.model_name, help="Hugging Face model (default: %(default)s)")
    parser.add_argument("--max-len", type=int, default=default.max_len, help="Max tokens (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=default.batch_size, help="Encoding batch size (default: %(default)s)")
    parser.add_argument("--allow-cpu", action="store_true", help="Run on CPU if no GPU is available (slow).")
    parser.add_argument("--carbon-json", default="results/carbon/upstream_embeddings.json",
                        help="Where to write the upstream time/energy/CO2 summary (default: %(default)s)")
    a = parser.parse_args()
    cfg = Config(**{"input_csv": a.input_csv, "output_emb": a.output_emb, "text_col": a.text_col, "model_name": a.model_name, "max_len": a.max_len, "batch_size": a.batch_size, "require_cuda": not a.allow_cpu})
    return cfg, a.carbon_json


def write_carbon_json(path: str, **record) -> None:
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Upstream resource summary written to {path}")


def main() -> None:
    cfg, carbon_json = parse_args()
    t_total0 = time.perf_counter()

    device = get_device(cfg.require_cuda)
    df = load_input_dataframe(cfg.input_csv, cfg.text_col)
    texts, ids = build_texts_and_ids(df, cfg.text_col)

    n = len(texts)
    if n == 0:
        raise ValueError(f"No rows found in {cfg.input_csv}")

    print(f"Loaded {n:,} rows from {cfg.input_csv}")

    tokenizer, model = build_model_and_tokenizer(cfg.model_name, device)
    warmup_model(model, tokenizer, device, cfg.max_len)

    emb_dict, encoding_time_s, energy_kwh, co2_g = run_encoding(
        texts=texts,
        ids=ids,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=cfg.batch_size,
        max_len=cfg.max_len,
    )

    save_embeddings(emb_dict, cfg.output_emb)

    total_time_s = time.perf_counter() - t_total0
    print_summary(
        n_rows=len(emb_dict),
        encoding_time_s=encoding_time_s,
        total_time_s=total_time_s,
        energy_kwh=energy_kwh,
        co2_g=co2_g,
        output_emb=cfg.output_emb,
    )
    write_carbon_json(
        carbon_json, stage="roberta_embeddings", model=cfg.model_name, n_items=len(emb_dict),
        inference_time_s=encoding_time_s, energy_kWh=energy_kwh, co2_kg=co2_g / 1000.0,
    )


if __name__ == "__main__":
    main()