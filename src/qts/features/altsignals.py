"""Anomaly providers over alt_signals daily counts. FORWARD_ONLY: crawl-time data
has no backfillable history, so these are excluded from backtests by leak guard #6
and enter live ranking only via the IC gate (config.DEFAULT_LIVE_GATED) once proven.

Direction LOWER_BETTER is PROVISIONAL (attention-spike ~ short-horizon bearish,
Da et al. 2011); the live IC record decides the real sign before any un-gating.
Zero-fill only within the globally-observed collection span — days before
collection started are unknown, not zero. z = (recent mean - base mean) /
max(base std, 1.0), clipped to [-10, 10] — a 1.0 count/day denominator floor
(not epsilon) and an explicit cap keep flat-base (std=0) symbols finite."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from .base import FeatureProvider, register

MIN_HISTORY_DAYS = 21
RECENT_DAYS = 7
BASE_DAYS = 63


def _anomaly(store: PITStore, symbols: list[str], as_of: date, concept: str) -> pd.Series:
    start = as_of - timedelta(days=BASE_DAYS + RECENT_DAYS)
    df = store.alt_signals_window(symbols, concept, start, as_of)
    out = pd.Series(np.nan, index=symbols)
    if df.empty:
        return out
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    span_start = max(start, df["dt"].min())        # collection start, not epoch
    span = pd.date_range(span_start, as_of, freq="D").date
    if len(span) < MIN_HISTORY_DAYS:
        return out
    cut = as_of - timedelta(days=RECENT_DAYS)
    piv = (df.pivot_table(index="dt", columns="symbol", values="value", aggfunc="sum")
             .reindex(span).fillna(0.0))
    base = piv[piv.index < cut]
    recent = piv[piv.index >= cut]
    if len(base) < MIN_HISTORY_DAYS - RECENT_DAYS or recent.empty:
        return out
    # Denominator floored at 1.0 (not epsilon): std()==0 is common for quiet,
    # zero-filled symbols, and an epsilon denominator blows a single extra
    # count up to z~1e9 -- which, because the ranker's cross-sectional
    # winsorization computes mean/std BEFORE clipping, silently destroys every
    # other symbol's z-score for the day. A floor of 1.0 count/day is a
    # meaningful scale for daily count data (Poisson-ish) and keeps z finite.
    z = (recent.mean() - base.mean()) / base.std().fillna(0.0).clip(lower=1.0)
    z = z.clip(lower=-10.0, upper=10.0)  # belt-and-braces cap on raw z
    # symbols with observed rows get a value; never-observed symbols stay NaN
    seen = set(df["symbol"])
    for s in symbols:
        if s in seen and s in z.index and pd.notna(z[s]):
            out[s] = float(z[s])
    return out


class NewsVolumeAnomaly(FeatureProvider):
    spec = FeatureSpec(
        name="news_volume_anomaly", direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.FORWARD_ONLY,
        description="z of 7d news-item count vs own trailing 63d (crawl-time PIT).")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        return _anomaly(store, symbols, as_of, "news_count")


class RedditMentionAnomaly(FeatureProvider):
    spec = FeatureSpec(
        name="reddit_mention_anomaly", direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.FORWARD_ONLY,
        description="z of 7d reddit mention count vs own trailing 63d (crawl-time PIT).")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        return _anomaly(store, symbols, as_of, "reddit_count")


register(NewsVolumeAnomaly())
register(RedditMentionAnomaly())
