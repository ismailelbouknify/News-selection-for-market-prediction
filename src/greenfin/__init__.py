"""GreenFin: resource-aware news selection for sustainable financial market direction forecasting.

Public names are imported lazily so that light-weight modules (e.g.
:mod:`greenfin.trading`, :mod:`greenfin.bootstrap`) can be used without
importing PyTorch.
"""
from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "Config": "config",
    "FININevaluation": "cv",
    "kfold_time_cv": "cv",
    "window_checkpoint_path": "cv",
    "GreenFinDataset": "dataset",
    "Sample": "dataset",
    "clone_dataset": "dataset",
    "make_splits": "dataset",
    "to_samples": "dataset",
    "evaluate": "evaluate",
    "evaluate_always_buy": "evaluate",
    "predict": "evaluate",
    "summarize_predictions": "evaluate",
    "configure_environment": "io",
    "load_embeddings": "io",
    "load_jsonl": "io",
    "sanity_check_embedding_coverage": "io",
    "set_seed": "io",
    "compute_pnl_sharpe": "metrics",
    "get_returns_map": "metrics",
    "GreenFin": "model",
    "wrap_model_for_multi_gpu": "model",
    "preselect_news": "selection",
    "select_farthest_ids": "selection",
    "select_kmeans_ids": "selection",
    "build_dataloaders": "train",
    "train_loop": "train",
    "build_fold_windows": "windows",
    "fold_date_splits": "windows",
    "split_indices": "windows",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'greenfin' has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)
