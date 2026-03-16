from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP


class GreenFin(nn.Module):
    def __init__(
        self,
        e_in: int = 768,
        v1: int = 64,
        v2: int = 32,
        v3: int = 128,
        v4: int = 128,
        mlp_layers: int = 2,
        mlp_hidden: int = 64,
        attn_heads: int = 4,
        attn_head_dim: Optional[int] = None,
        temporal: str = "lstm",
        t: int = 5,
        p_dropout: float = 0.05,
        use_market: bool = True,
        use_news: bool = True,
        use_sentiment: bool = True,
        use_miq: bool = True,
        n_market_feats: int = 5,
    ) -> None:
        super().__init__()

        if not use_news:
            use_sentiment = False

        self.use_market = use_market
        self.use_news = use_news
        self.use_sentiment = use_sentiment
        self.use_miq = use_miq

        if self.use_market:
            self.e1_market = MLP(n_market_feats, v1, mlp_hidden, mlp_layers, p_dropout)
            self.norm_m = nn.LayerNorm(v1)
        else:
            v1 = 0

        if self.use_news:
            if attn_head_dim is not None:
                v4 = attn_heads * attn_head_dim

            if self.use_sentiment:
                self.e2_sent = MLP(3, v2, mlp_hidden, mlp_layers, p_dropout)
            else:
                v2 = 0
            self.e3_head = MLP(e_in, v3, mlp_hidden, mlp_layers, p_dropout)
            self.fuse = MLP(v2 + v3, v4, mlp_hidden, mlp_layers, p_dropout)
            self.norm_r = nn.LayerNorm(v4)

            if self.use_miq:
                assert v4 % attn_heads == 0, f"v4 ({v4}) must be divisible by attn_heads ({attn_heads})."
                self.self_attn = nn.MultiheadAttention(v4, attn_heads, dropout=p_dropout, batch_first=True)
                self.cross_attn = nn.MultiheadAttention(v4, attn_heads, dropout=p_dropout, batch_first=True)
                self.q_proj = nn.Linear(max(1, v1), v4) if self.use_market else nn.Identity()
        else:
            v2 = 0
            v3 = 0
            v4 = 0

        self.dim_market = v1 if self.use_market else 0
        self.dim_fused = v4 if self.use_news else 0
        d_day = self.dim_market + self.dim_fused
        if d_day == 0:
            raise ValueError("At least one modality must be enabled (market or news).")
        self.d_day = d_day

        if temporal == "lstm":
            self.tem = nn.LSTM(d_day, max(64, d_day), 1, batch_first=True)
            self.tem_out = nn.Linear(max(64, d_day), 1)
        elif temporal == "mlp":
            self.tem = MLP(d_day * t, max(128, d_day), mlp_hidden, mlp_layers, p_dropout)
            self.tem_out = nn.Linear(max(128, d_day), 1)
        elif temporal == "cnn1d":
            c = max(32, d_day // 2)
            self.tem_conv = nn.Conv1d(d_day, c, 3, padding=1)
            self.tem_out = nn.Linear(c * t, 1)
        else:
            raise ValueError("temporal must be lstm/mlp/cnn1d")

    def forward(
        self,
        markets: torch.Tensor,
        news_emb: Optional[torch.Tensor] = None,
        sentiments: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, t = markets.shape[0], markets.shape[1]
        attn_weights = None

        if self.use_market:
            m = self.norm_m(self.e1_market(markets))
        else:
            m = torch.zeros(bsz, t, 0, device=markets.device)

        use_news_now = self.use_news and news_emb is not None and news_emb.shape[2] > 0

        if use_news_now:
            s = (
                self.e2_sent(sentiments)
                if self.use_sentiment
                else torch.zeros(bsz, t, news_emb.shape[2], 0, device=markets.device)
            )
            hh = self.e3_head(news_emb)
            r = self.norm_r(self.fuse(torch.cat([s, hh], dim=-1)))

            if self.use_miq:
                r_flat = r.reshape(bsz * t, r.shape[2], -1)
                mask_flat = pad_mask.reshape(bsz * t, r.shape[2])
                r_ctx, _ = self.self_attn(r_flat, r_flat, r_flat, key_padding_mask=mask_flat)

                if self.use_market:
                    q = self.q_proj(m).reshape(bsz * t, 1, -1)
                    fd, attn_weights = self.cross_attn(q, r_ctx, r_ctx, key_padding_mask=mask_flat)
                    fd = fd.reshape(bsz, t, -1)
                else:
                    fd = r_ctx.mean(dim=1).reshape(bsz, t, -1)
                    attn_weights = None
            else:
                valid = (~pad_mask).float()
                denom = valid.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
                fd = (r * valid.unsqueeze(-1)).sum(dim=2) / denom

            x = torch.cat([m, fd], dim=-1) if m.numel() else fd
        else:
            if self.dim_fused > 0:
                zeros_news = torch.zeros(
                    bsz,
                    t,
                    self.dim_fused,
                    device=markets.device,
                    dtype=m.dtype if m.numel() else torch.float32,
                )
                x = torch.cat([m, zeros_news], dim=-1) if m.numel() else zeros_news
            else:
                x = m

        if hasattr(self, "tem_conv"):
            feat = F.relu(self.tem_conv(x.permute(0, 2, 1)))
            logits = self.tem_out(feat.reshape(bsz, -1))
        elif isinstance(self.tem, nn.LSTM):
            out, _ = self.tem(x)
            logits = self.tem_out(out[:, -1, :])
        else:
            logits = self.tem_out(self.tem(x.reshape(bsz, -1)))

        return logits, attn_weights


def wrap_model_for_multi_gpu(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            print(f"[multi-gpu] Using DataParallel on {n} GPUs.")
            model = nn.DataParallel(model)
        else:
            print("[multi-gpu] Single GPU detected; running normally.")
    else:
        print("[multi-gpu] CUDA not available; running on CPU.")
    return model
