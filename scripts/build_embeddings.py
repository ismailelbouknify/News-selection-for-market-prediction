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
            "Set REQUIRE_CUDA=False if you want to allow CPU encoding."
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


def main() -> None:
    cfg = Config()
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


if __name__ == "__main__":
    main()