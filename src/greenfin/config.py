from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # training
    epochs: int = 10
    batch_size: int = 16
    lr: float = 2e-4
    amp: bool = False
    seed: int = 2
    early_stop_patience: int = 100

    # model
    v1: int = 32
    v2: int = 16
    v3: int = 32
    v4: int = 32
    mlp_layers: int = 2
    mlp_hidden: int = 64
    attn_heads: int = 3
    attn_head_dim: Optional[int] = 32
    temporal: str = "mlp"  # "mlp" | "lstm" | "cnn1d"

    # task
    decision_threshold: float = 0.5
    rf_annual: float = 0.02
    use_market: bool = True
    use_news: bool = True
    use_sentiment: bool = True
    use_miq: bool = True
    loss_name: str = "bce"
    focal_alpha: float = 1.0
    focal_gamma: float = 3.0

    # data / selection / checkpoints
    cap_per_day: Optional[int] = None
    news_select: Optional[str] = None  # "kmeans" | "topconf" | "random" | "farthest"
    best_model_path: str = "best_model.pt"
    best_select: str = "val_pnl"  # "val_pnl" | "val_sharpe" | "val_acc" | "val_loss"
    use_best_for_test: bool = True

    # logging
    carbon_log_dir: str = "carbon_logs"
