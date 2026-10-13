"""Decayed sentiment providers over alt_signals daily (sum, n) sentiment concepts.
FORWARD_ONLY (crawl/score-time data, no backfillable history) and LIVE-GATED until
the IC record earns un-gating — same contract as features/altsignals.py.

value(symbol, as_of) = sum_d w_d * sent_sum_d / sum_d w_d * sent_n_d over the last
LOOKBACK_DAYS, w_d = 0.5 ** (age_days / HALF_LIFE_DAYS). A mention-weighted mean
with exponential decay: fresh chatter dominates, a lone stale burst fades. NaN when
fewer than MIN_MENTIONS mentions land in the window (too thin to mean anything)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from .base import FeatureProvider, register

HALF_LIFE_DAYS = 5.0
LOOKBACK_DAYS = 21
MIN_MENTIONS = 3


def _decayed_sentiment(store: PITStore, symbols: list[str], as_of: date,
                       prefix: str) -> pd.Series:
    start = as_of - timedelta(days=LOOKBACK_DAYS)
    sums = store.alt_signals_window(symbols, f"{prefix}_sum", start, as_of)
    ns = store.alt_signals_window(symbols, f"{prefix}_n", start, as_of)
    out = pd.Series(np.nan, index=symbols)
    if sums.empty or ns.empty:
        return out
    for df in (sums, ns):
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
        df["w"] = [0.5 ** ((as_of - d).days / HALF_LIFE_DAYS) for d in df["dt"]]
    wsum = (sums["value"] * sums["w"]).groupby(sums["symbol"]).sum()
    wn = (ns["value"] * ns["w"]).groupby(ns["symbol"]).sum()
    n_total = ns.groupby("symbol")["value"].sum()
    for s in symbols:
        if n_total.get(s, 0.0) >= MIN_MENTIONS and wn.get(s, 0.0) > 0:
            out[s] = float(wsum.get(s, 0.0) / wn[s])
    return out


class RedditSentiment(FeatureProvider):
    spec = FeatureSpec(
        name="reddit_sentiment", direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.FORWARD_ONLY,
        description="Decayed mention-weighted mean reddit sentiment (GPU worker scored).")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        return _decayed_sentiment(store, symbols, as_of, "reddit_sent")


class NewsSentiment(FeatureProvider):
    spec = FeatureSpec(
        name="news_sentiment", direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.FORWARD_ONLY,
        description="Decayed mention-weighted mean news sentiment (GPU worker scored).")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        return _decayed_sentiment(store, symbols, as_of, "news_sent")


register(RedditSentiment())
register(NewsSentiment())
