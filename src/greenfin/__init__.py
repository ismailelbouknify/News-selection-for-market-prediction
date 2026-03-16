
__all__ = [
    "Config",
    "FININevaluation",
    "kfold_time_cv",
    "GreenFinDataset",
    "Sample",
    "clone_dataset",
    "make_splits",
    "to_samples",
    "evaluate",
    "evaluate_always_buy",
    "configure_environment",
    "load_embeddings",
    "load_jsonl",
    "sanity_check_embedding_coverage",
    "set_seed",
    "GreenFin",
    "wrap_model_for_multi_gpu",
    "preselect_news",
    "select_farthest_ids",
    "select_kmeans_ids",
    "build_dataloaders",
    "train_loop",
]


def __getattr__(name):
    if name == "Config":
        from .config import Config
        return Config

    if name in {"FININevaluation", "kfold_time_cv"}:
        from .cv import FININevaluation, kfold_time_cv
        return {"FININevaluation": FININevaluation, "kfold_time_cv": kfold_time_cv}[name]

    if name in {"GreenFinDataset", "Sample", "clone_dataset", "make_splits", "to_samples"}:
        from .dataset import GreenFinDataset, Sample, clone_dataset, make_splits, to_samples
        return {
            "GreenFinDataset": GreenFinDataset,
            "Sample": Sample,
            "clone_dataset": clone_dataset,
            "make_splits": make_splits,
            "to_samples": to_samples,
        }[name]

    if name in {"evaluate", "evaluate_always_buy"}:
        from .evaluate import evaluate, evaluate_always_buy
        return {"evaluate": evaluate, "evaluate_always_buy": evaluate_always_buy}[name]

    if name in {"configure_environment", "load_embeddings", "load_jsonl", "sanity_check_embedding_coverage", "set_seed"}:
        from .io import configure_environment, load_embeddings, load_jsonl, sanity_check_embedding_coverage, set_seed
        return {
            "configure_environment": configure_environment,
            "load_embeddings": load_embeddings,
            "load_jsonl": load_jsonl,
            "sanity_check_embedding_coverage": sanity_check_embedding_coverage,
            "set_seed": set_seed,
        }[name]

    if name in {"GreenFin", "wrap_model_for_multi_gpu"}:
        from .model import GreenFin, wrap_model_for_multi_gpu
        return {
            "GreenFin": GreenFin,
            "wrap_model_for_multi_gpu": wrap_model_for_multi_gpu,
        }[name]

    if name in {"preselect_news", "select_farthest_ids", "select_kmeans_ids"}:
        from .selection import preselect_news, select_farthest_ids, select_kmeans_ids
        return {
            "preselect_news": preselect_news,
            "select_farthest_ids": select_farthest_ids,
            "select_kmeans_ids": select_kmeans_ids,
        }[name]

    if name in {"build_dataloaders", "train_loop"}:
        from .train import build_dataloaders, train_loop
        return {
            "build_dataloaders": build_dataloaders,
            "train_loop": train_loop,
        }[name]

    raise AttributeError(f"module 'greenfin' has no attribute {name!r}")