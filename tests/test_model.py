from __future__ import annotations

import torch

from greenfin.model import GreenFin


def test_market_only_forward():
    model = GreenFin(
        e_in=8,
        temporal="mlp",
        t=2,
        use_market=True,
        use_news=False,
        use_sentiment=False,
        use_miq=False,
        n_market_feats=5,
        v1=8,
        v2=4,
        v3=8,
        v4=8,
        mlp_layers=2,
        mlp_hidden=16,
        attn_heads=2,
        attn_head_dim=4,
    )

    markets = torch.randn(3, 2, 5)
    logits, attn = model(markets)

    assert logits.shape == (3, 1)
    assert attn is None


def test_market_and_news_forward_without_miq():
    model = GreenFin(
        e_in=8,
        temporal="mlp",
        t=2,
        use_market=True,
        use_news=True,
        use_sentiment=True,
        use_miq=False,
        n_market_feats=5,
        v1=8,
        v2=4,
        v3=8,
        v4=8,
        mlp_layers=2,
        mlp_hidden=16,
        attn_heads=2,
        attn_head_dim=4,
    )

    markets = torch.randn(2, 2, 5)
    news_emb = torch.randn(2, 2, 3, 8)
    sentiments = torch.randn(2, 2, 3, 3)
    pad_mask = torch.tensor(
        [
            [[False, False, True], [False, True, True]],
            [[False, False, False], [False, False, True]],
        ],
        dtype=torch.bool,
    )

    logits, attn = model(markets, news_emb, sentiments, pad_mask)

    assert logits.shape == (2, 1)
    assert attn is None


def test_market_and_news_forward_with_miq_returns_attention():
    model = GreenFin(
        e_in=8,
        temporal="mlp",
        t=2,
        use_market=True,
        use_news=True,
        use_sentiment=True,
        use_miq=True,
        n_market_feats=5,
        v1=8,
        v2=4,
        v3=8,
        v4=8,
        mlp_layers=2,
        mlp_hidden=16,
        attn_heads=2,
        attn_head_dim=4,
    )

    markets = torch.randn(2, 2, 5)
    news_emb = torch.randn(2, 2, 3, 8)
    sentiments = torch.randn(2, 2, 3, 3)
    pad_mask = torch.zeros(2, 2, 3, dtype=torch.bool)

    logits, attn = model(markets, news_emb, sentiments, pad_mask)

    assert logits.shape == (2, 1)
    assert attn is not None


def test_model_handles_missing_news_batch_even_when_news_is_enabled():
    model = GreenFin(
        e_in=8,
        temporal="mlp",
        t=2,
        use_market=True,
        use_news=True,
        use_sentiment=True,
        use_miq=True,
        n_market_feats=5,
        v1=8,
        v2=4,
        v3=8,
        v4=8,
        mlp_layers=2,
        mlp_hidden=16,
        attn_heads=2,
        attn_head_dim=4,
    )

    markets = torch.randn(4, 2, 5)

    # news_emb=None should trigger the "no news now" branch
    logits, attn = model(markets, news_emb=None, sentiments=None, pad_mask=None)

    assert logits.shape == (4, 1)
    assert attn is None
