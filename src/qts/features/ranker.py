"""Cross-sectional composite ranker.

V1 = winsorized, sign-adjusted z-score composite (transparent, no overfit on short
history / small universe). Optional IC weights (fed only past-realized data by the
backtest) and a LightGBM seam are left for when breadth/history justify them.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Direction
from .base import FeatureProvider


def compute_features(
    store: PITStore, providers: list[FeatureProvider], symbols: list[str], as_of: date,
) -> pd.DataFrame:
    cols = {}
    for p in providers:
        try:
            cols[p.spec.name] = p.compute(store, symbols, as_of)
        except Exception as e:  # noqa: BLE001
            cols[p.spec.name] = pd.Series(np.nan, index=symbols)
    return pd.DataFrame(cols, index=symbols)


def _winsor_z(s: pd.Series, k: float = 2.5) -> pd.Series:
    s = s.astype(float)
    if s.notna().sum() < 3:
        return pd.Series(np.nan, index=s.index)
    mu, sd = s.mean(), s.std()
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index).where(s.notna())
    z = (s - mu) / sd
    return z.clip(-k, k)


def cross_section_zscore(
    feats: pd.DataFrame, providers: list[FeatureProvider],
) -> pd.DataFrame:
    """Per-feature winsorized z-score, sign-adjusted so HIGHER = more bullish.
    All-NaN columns (e.g. unpopulated short_interest) are dropped → self-excluding."""
    dirn = {p.spec.name: p.spec.direction for p in providers}
    out = {}
    for col in feats.columns:
        if feats[col].notna().sum() < 3:
            continue  # not enough cross-section -> exclude this feature this period
        z = _winsor_z(feats[col])
        if dirn.get(col) == Direction.LOWER_BETTER:
            z = -z
        out[col] = z
    return pd.DataFrame(out, index=feats.index)


def composite_score(zdf: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    if zdf.empty:
        return pd.Series(dtype=float)
    if weights:
        w = pd.Series({c: weights.get(c, 0.0) for c in zdf.columns})
        w = w / (w.abs().sum() or 1.0)
        score = (zdf * w).sum(axis=1, min_count=1)
    else:
        score = zdf.mean(axis=1)  # equal weight over available features
    return score


def assign_buckets(score: pd.Series, n: int = 5) -> pd.Series:
    """1 = worst (short), n = best (long). NaN scores excluded."""
    s = score.dropna()
    if len(s) < n:
        return pd.Series(np.nan, index=score.index)
    try:
        b = pd.qcut(s.rank(method="first"), n, labels=range(1, n + 1))
    except ValueError:
        return pd.Series(np.nan, index=score.index)
    return b.reindex(score.index).astype("float")


def rank_universe(
    store: PITStore, providers: list[FeatureProvider], symbols: list[str],
    as_of: date, n_buckets: int = 5, weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Full pipeline at one as_of -> DataFrame[symbol, score, bucket, <z-cols>]."""
    feats = compute_features(store, providers, symbols, as_of)
    zdf = cross_section_zscore(feats, providers)
    score = composite_score(zdf, weights)
    # Eligibility: a name needs real price history to be tradeable. Event providers emit
    # 0.0 for every symbol, so without this gate a data-less name (e.g. a delisted ticker
    # with no ingested prices) would still receive a score/bucket. Require price coverage.
    price_cols = [c for c in ("momentum_12_1", "low_volatility") if c in feats]
    if price_cols:
        eligible = feats[price_cols].notna().any(axis=1)
        score = score.where(eligible)
    bucket = assign_buckets(score, n_buckets)
    out = pd.DataFrame({"score": score, "bucket": bucket})
    out = out.join(zdf.add_prefix("z_")).join(feats.add_prefix("raw_"))
    return out.sort_values("score", ascending=False)
