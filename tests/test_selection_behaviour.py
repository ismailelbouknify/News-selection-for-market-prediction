from __future__ import annotations

import numpy as np
import pytest
import torch

from greenfin.dataset import GreenFinDataset
from greenfin.selection import preselect_news, select_farthest_ids, select_kmeans_ids


def _one_day_dataset(n_headlines: int, sentiments=None):
    ids = list(range(1, n_headlines + 1))
    if sentiments is None:
        rng = np.random.default_rng(0)
        p = rng.dirichlet([1, 1, 1], size=n_headlines)
        sentiments = p.tolist()
    return GreenFinDataset([{
        "date": "2024-01-02", "markets": [[1, 1, 1, 1, 1]],
        "headline_ids": [ids], "sentiments": [sentiments], "label": 1,
    }])


def _emb(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return {i: torch.tensor(rng.normal(size=dim), dtype=torch.float16) for i in range(1, n + 1)}


@pytest.mark.parametrize("mode", ["random", "topconf", "kmeans", "farthest"])
def test_every_selector_returns_exactly_k_when_enough_headlines(mode):
    ds = _one_day_dataset(30)
    preselect_news(ds, cap_per_day=10, selection_mode=mode, seed=1, emb_dict=_emb(30))
    ids = ds.samples[0].headline_ids[0]
    assert len(ids) == 10 and len(set(ids)) == 10
    assert len(ds.samples[0].sentiments[0]) == 10


@pytest.mark.parametrize("mode", ["random", "topconf", "kmeans", "farthest"])
def test_days_with_fewer_than_k_headlines_are_kept_unchanged(mode):
    ds = _one_day_dataset(4)
    before = list(ds.samples[0].headline_ids[0])
    preselect_news(ds, cap_per_day=10, selection_mode=mode, seed=1, emb_dict=_emb(4))
    assert ds.samples[0].headline_ids[0] == before


@pytest.mark.parametrize("mode", ["random", "kmeans", "farthest"])
def test_selection_is_deterministic_given_seed(mode):
    emb = _emb(40)
    a, b = _one_day_dataset(40), _one_day_dataset(40)
    preselect_news(a, cap_per_day=7, selection_mode=mode, seed=11, emb_dict=emb)
    preselect_news(b, cap_per_day=7, selection_mode=mode, seed=11, emb_dict=emb)
    assert a.samples[0].headline_ids == b.samples[0].headline_ids


def test_random_selection_depends_on_seed():
    picks = set()
    for seed in range(5):
        ds = _one_day_dataset(40)
        preselect_news(ds, cap_per_day=5, selection_mode="random", seed=seed, emb_dict=None)
        picks.add(tuple(ds.samples[0].headline_ids[0]))
    assert len(picks) > 1


def test_topconf_keeps_lowest_entropy_headlines_in_original_order():
    sents = [
        [1 / 3, 1 / 3, 1 / 3],   # id 1: max entropy
        [0.98, 0.01, 0.01],      # id 2: very confident
        [0.5, 0.25, 0.25],       # id 3
        [0.01, 0.98, 0.01],      # id 4: very confident
        [0.4, 0.3, 0.3],         # id 5
    ]
    ds = _one_day_dataset(5, sentiments=sents)
    preselect_news(ds, cap_per_day=2, selection_mode="topconf", seed=0, emb_dict=None)
    assert ds.samples[0].headline_ids[0] == [2, 4]
    assert ds.samples[0].sentiments[0] == [sents[1], sents[3]]


def test_farthest_point_prefers_diverse_headlines():
    # Two tight clusters of near-duplicates plus one far-away headline.
    base_a, base_b, far = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2]
    vecs = [base_a, base_a + 0.01, base_a + 0.02, base_b, base_b + 0.01, far]
    emb = {i + 1: torch.tensor(v, dtype=torch.float32) for i, v in enumerate(vecs)}
    chosen = select_farthest_ids(list(emb), emb, k=3)
    clusters = {1: "a", 2: "a", 3: "a", 4: "b", 5: "b", 6: "far"}
    assert sorted(clusters[c] for c in chosen) == ["a", "b", "far"]


def test_kmeans_and_farthest_skip_ids_without_embeddings():
    emb = _emb(10)
    ids = list(range(1, 13))  # 11 and 12 have no embedding
    assert set(select_kmeans_ids(ids, [], emb, k=4)) <= set(emb)
    assert set(select_farthest_ids(ids, emb, k=4)) <= set(emb)
    assert len(select_kmeans_ids(ids, [], emb, k=4)) == 4
