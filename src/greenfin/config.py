from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TEMPORAL_BACKBONES = ("mlp", "lstm", "cnn1d")
SELECTION_MODES = ("kmeans", "topconf", "random", "farthest")
BEST_SELECT = ("val_pnl", "val_sharpe", "val_acc", "val_loss")
LOSSES = ("bce", "focal")


@dataclass
class Config:
    """Runtime configuration of one experiment.

    Defaults are the library defaults inherited from the research code. The
    values actually used in the paper are set in ``configs/base.yaml``
    (``epochs=50``, ``batch_size=4``, ``lr=1e-3``, ``grad_accum_steps=16``).

    Architecture switches:

    * ``use_miq=True``  -> FININ-style market-aware interaction (self-attention
      over a day's headlines + market-query cross-attention);
    * ``use_miq=False`` -> GreenFin masked-mean pooling over a day's headlines;
    * ``use_news=False`` -> market-only model.
    """

    experiment_name: str = "experiment"

    # training
    epochs: int = 10
    batch_size: int = 16
    lr: float = 2e-4
    grad_accum_steps: int = 16
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
    best_model_path: str = "best_model.pt"  # base name; one file per (seed, window) is derived from it
    best_select: str = "val_pnl"  # "val_pnl" | "val_sharpe" | "val_acc" | "val_loss"
    use_best_for_test: bool = True

    # resource tracking
    track_carbon: bool = True
    carbon_save_to_file: bool = False
    carbon_log_dir: str = "carbon_logs"

    def __post_init__(self) -> None:
        if self.temporal not in TEMPORAL_BACKBONES:
            raise ValueError(f"temporal must be one of {TEMPORAL_BACKBONES}, got {self.temporal!r}")
        if self.news_select is not None and self.news_select not in SELECTION_MODES:
            raise ValueError(f"news_select must be one of {SELECTION_MODES} or null, got {self.news_select!r}")
        if self.cap_per_day is not None and int(self.cap_per_day) < 1:
            raise ValueError(f"cap_per_day must be a positive integer or null, got {self.cap_per_day!r}")
        if self.cap_per_day is not None and self.news_select is None:
            raise ValueError("cap_per_day is set but news_select is null; choose a selector.")
        if self.best_select not in BEST_SELECT:
            raise ValueError(f"best_select must be one of {BEST_SELECT}, got {self.best_select!r}")
        if self.loss_name not in LOSSES:
            raise ValueError(f"loss_name must be one of {LOSSES}, got {self.loss_name!r}")
        if not (self.use_market or self.use_news):
            raise ValueError("At least one of use_market / use_news must be true.")
        for name in ("epochs", "batch_size", "grad_accum_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)!r}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr!r}")
