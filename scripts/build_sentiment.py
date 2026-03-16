from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =======================
# Config
# =======================
INPUT_CSV = "data/interim/cleaned_news.csv"
OUT_CSV = "data/interim/cleaned_news_sentiment.csv"
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN = 512
BATCH_SIZE = 3072  
NUM_WORKERS = 0
PIN_MEMORY = True
REQUIRE_CUDA = True
# =======================


@dataclass(frozen=True)
class Config:
    input_csv: str = INPUT_CSV
    output_csv: str = OUT_CSV
    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    require_cuda: bool = REQUIRE_CUDA


class NewsTextDataset(Dataset):
    """Tokenizes financial news text for FinBERT inference."""

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def build_texts(df: pd.DataFrame) -> pd.Series:
    """Combine title and summary into a single inference text."""
    title = df["Article_title"].fillna("").astype(str)
    summary = df["Lsa_summary"].fillna("").astype(str)
    return (title + ". " + summary).str.strip()


def load_dataframe(input_csv: str) -> tuple[pd.DataFrame, List[str]]:
    """Load the input CSV and keep only rows with non-empty combined text."""
    df = pd.read_csv(input_csv)

    required_cols = {"Article_title", "Lsa_summary"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    texts_series = build_texts(df)
    mask = texts_series.str.len() > 0

    filtered_df = df.loc[mask].copy().reset_index(drop=True)
    texts = texts_series.loc[mask].tolist()

    return filtered_df, texts


def get_device(require_cuda: bool) -> torch.device:
    """Select the execution device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_cuda:
        raise RuntimeError(
            "CUDA GPU required for this script. "
            "Set REQUIRE_CUDA=False if you want to allow CPU inference."
        )

    return torch.device("cpu")


def build_model_and_tokenizer(model_name: str, device: torch.device) -> tuple[AutoTokenizer, torch.nn.Module]:
    """Load FinBERT tokenizer and sequence classification model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for inference")

    model = model.to(device).eval()
    return tokenizer, model


def warmup_model(model: torch.nn.Module, device: torch.device, max_len: int) -> None:
    """Optional GPU warmup for more stable timing."""
    if device.type != "cuda":
        return

    with torch.no_grad():
        torch.cuda.synchronize()
        _ = model(
            input_ids=torch.zeros((1, max_len), dtype=torch.long, device=device),
            attention_mask=torch.ones((1, max_len), dtype=torch.long, device=device),
        )
        torch.cuda.synchronize()


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, List[int], float, float, float]:
    """
    Run FinBERT inference and return:
      - logits array of shape [N, C]
      - predicted class indices
      - inference time in seconds
      - energy in kWh
      - CO2 in grams
    """
    tracker = EmissionsTracker(save_to_file=False, log_level="error")

    all_logits: List[torch.Tensor] = []
    all_pred_ids: List[int] = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    tracker.start()

    try:
        for batch in tqdm(dataloader, desc="FinBERT inference"):
            input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
            attention_mask = batch["attention_mask"].to(device, non_blocking=(device.type == "cuda"))

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            all_logits.append(logits.detach().cpu())
            preds = torch.argmax(logits, dim=1)
            all_pred_ids.extend(preds.detach().cpu().tolist())
    finally:
        co2_kg = tracker.stop()
        if device.type == "cuda":
            torch.cuda.synchronize()

    inference_time_s = time.perf_counter() - t0

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

    logits_array = torch.cat(all_logits, dim=0).numpy() if all_logits else np.empty((0, 3), dtype=np.float32)
    return logits_array, all_pred_ids, inference_time_s, energy_kwh, co2_g


def get_id2label(model: torch.nn.Module) -> Dict[int, str]:
    """Extract id2label mapping from a plain model or DataParallel wrapper."""
    cfg = model.module.config if isinstance(model, torch.nn.DataParallel) else model.config
    return {int(k): str(v).lower() for k, v in cfg.id2label.items()}


def attach_outputs(
    df: pd.DataFrame,
    logits_array: np.ndarray,
    pred_ids: List[int],
    id2label: Dict[int, str],
) -> pd.DataFrame:
    """
    Add one column per class plus predicted_label.

    Important:
    We intentionally save raw logits, not probabilities, because your
    downstream dataset builder applies softmax later.
    """
    out = df.copy()

    if logits_array.shape[0] != len(out):
        raise ValueError(
            f"Row mismatch: dataframe has {len(out)} rows but logits have {logits_array.shape[0]} rows."
        )

    for idx in sorted(id2label):
        class_name = id2label[idx]
        out[class_name] = logits_array[:, idx]

    out["predicted_label"] = [id2label[i] for i in pred_ids]
    return out


def print_summary(
    n_texts: int,
    inference_time_s: float,
    total_time_s: float,
    energy_kwh: float,
    co2_g: float,
) -> None:
    """Print runtime and sustainability metrics."""
    print("\nSaved sentiment file successfully.")
    print("---- Timing ----")
    print(f"Inference time: {inference_time_s:.2f} s  ({inference_time_s / 60.0:.2f} min)")
    print(f"Total time    : {total_time_s:.2f} s  ({total_time_s / 60.0:.2f} min)")
    print("---- CodeCarbon ----")
    print(f"Energy (kWh)  : {energy_kwh:.6f}")
    print(f"CO2 (g)       : {co2_g:.2f}")
    print("---- Per text ----")
    print(f"Texts         : {n_texts:,}")
    print(f"Time/text (ms): {(inference_time_s * 1000.0) / max(1, n_texts):.4f}")
    print(f"Energy/text (Wh): {(energy_kwh * 1000.0) / max(1, n_texts):.8f}")
    print(f"CO2/text (mg) : {(co2_g * 1000.0) / max(1, n_texts):.8f}")


def main() -> None:
    cfg = Config()
    t_total0 = time.perf_counter()

    df, texts = load_dataframe(cfg.input_csv)
    n = len(texts)
    print(f"Loaded {n:,} non-empty texts from {cfg.input_csv}")

    if n == 0:
        raise ValueError("No non-empty texts found after combining Article_title and Lsa_summary.")

    device = get_device(cfg.require_cuda)
    tokenizer, model = build_model_and_tokenizer(cfg.model_name, device)

    dataset = NewsTextDataset(texts=texts, tokenizer=tokenizer, max_length=cfg.max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
    )

    warmup_model(model, device, cfg.max_len)

    logits_array, pred_ids, inference_time_s, energy_kwh, co2_g = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    id2label = get_id2label(model)
    output_df = attach_outputs(df=df, logits_array=logits_array, pred_ids=pred_ids, id2label=id2label)
    output_df.to_csv(cfg.output_csv, index=False)

    total_time_s = time.perf_counter() - t_total0
    print_summary(
        n_texts=n,
        inference_time_s=inference_time_s,
        total_time_s=total_time_s,
        energy_kwh=energy_kwh,
        co2_g=co2_g,
    )


if __name__ == "__main__":
    main()