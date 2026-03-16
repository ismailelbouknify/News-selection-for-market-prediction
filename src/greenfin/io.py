from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


DEFAULT_ENV = {
    "TOKENIZERS_PARALLELISM": "false",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SHM_DISABLE": "0",
    "NCCL_DEBUG": "WARN",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}


def configure_environment() -> None:
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_embeddings(emb_path: str, fp16_source: bool = True) -> Dict[int, torch.Tensor]:
    emb_dict = torch.load(emb_path, map_location="cpu")
    out: Dict[int, torch.Tensor] = {}
    for k, v in emb_dict.items():
        t = v.float() if (fp16_source and getattr(v, "dtype", None) == torch.float16) else v
        out[int(k)] = t.view(-1)
    return out


def sanity_check_embedding_coverage(
    records: Sequence[Dict[str, Any]],
    emb_dict: Dict[int, torch.Tensor],
    max_days: int = 200,
) -> None:
    seen = have = days = empty_days = 0
    for r in records[:max_days]:
        days += 1
        ids_per_day = 0
        for day_ids in r.get("headline_ids", []):
            ids_per_day += len(day_ids)
            seen += len(day_ids)
            for hid in day_ids:
                if int(hid) in emb_dict:
                    have += 1
        if ids_per_day == 0:
            empty_days += 1
    cov = 100.0 * (have / max(1, seen)) if seen > 0 else 0.0
    print(
        f"[sanity] headline ids seen={seen}, with_embeddings={have} ({cov:.2f}%) "
        f"over first {days} samples; empty-news days={empty_days}"
    )


def set_seed(seed: int = 15) -> bool:
    import random

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        return False
    try:
        for dev in range(torch.cuda.device_count()):
            with torch.cuda.device(dev):
                g = torch.Generator(device="cuda")
                g.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return True
    except RuntimeError as e:
        print(f"[seed] CUDA seeding skipped: {e}")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        return False
