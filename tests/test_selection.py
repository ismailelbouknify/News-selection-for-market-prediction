from __future__ import annotations

from copy import deepcopy

import pytest

from greenfin.dataset import GreenFinDataset
from greenfin.selection import preselect_news


def _fresh_dataset(records):
    return GreenFinDataset(deepcopy(records))


def _assert_alignment(ds, cap):
    for sample in ds.samples:
        for ids_d, sents_d in zip(sample.headline_ids, sample.sentiments):
            assert len(ids_d) == len(sents_d)
            assert len(ids_d) <= cap


def test_preselect_news_topconf_caps_each_day(tiny_records):
    ds = _fresh_dataset(tiny_records)
    preselect_news(ds, cap_per_day=2, selection_mode="topconf", seed=123, emb_dict=None)

    _assert_alignment(ds, 2)


def test_preselect_news_random_preserves_empty_days(tiny_records):
    ds = _fresh_dataset(tiny_records)
    original_empty_day = ds.samples[0].headline_ids[1] == []

    preselect_news(ds, cap_per_day=1, selection_mode="random", seed=123, emb_dict=None)

    assert original_empty_day
    assert ds.samples[0].headline_ids[1] == []
    assert ds.samples[0].sentiments[1] == []
    _assert_alignment(ds, 1)


def test_preselect_news_kmeans_runs_with_embeddings(tiny_records, tiny_emb_dict):
    ds = _fresh_dataset(tiny_records)
    preselect_news(ds, cap_per_day=2, selection_mode="kmeans", seed=42, emb_dict=tiny_emb_dict)

    _assert_alignment(ds, 2)


def test_preselect_news_farthest_runs_with_embeddings(tiny_records, tiny_emb_dict):
    ds = _fresh_dataset(tiny_records)
    preselect_news(ds, cap_per_day=2, selection_mode="farthest", seed=42, emb_dict=tiny_emb_dict)

    _assert_alignment(ds, 2)


def test_preselect_news_noop_when_cap_is_none(tiny_records):
    ds = _fresh_dataset(tiny_records)
    before = deepcopy([(s.headline_ids, s.sentiments) for s in ds.samples])

    preselect_news(ds, cap_per_day=None, selection_mode="random", seed=42, emb_dict=None)

    after = [(s.headline_ids, s.sentiments) for s in ds.samples]
    assert before == after


def test_preselect_news_rejects_invalid_mode(tiny_records):
    ds = _fresh_dataset(tiny_records)

    with pytest.raises(ValueError, match="selection_mode must be one of"):
        preselect_news(ds, cap_per_day=2, selection_mode="invalid_mode", seed=42, emb_dict=None)
