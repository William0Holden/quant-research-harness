"""Store-backed computation of the three crypto sentiment signals, PIT.

Thin wrappers that pull daily attention/sentiment from `alt_signals` (the same table the
equity pipeline fills, keyed here by coin symbols) and hand plain Series to the pure
functions in signals.py. NaN-safe throughout: a signal with too little data returns "no
opinion", and the crypto book, while these are gated, does nothing with any of it.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ...pit.store import PITStore
from . import signals as sg

# Concepts summed into a coin's daily "attention" (mention volume). Same concept names the
# equity aggregation writes; crypto rows just carry coin symbols.
ATTENTION_CONCEPTS = ("news_count", "reddit_count")
SENT_CONCEPTS = (("news_sent_sum", "news_sent_n"), ("reddit_sent_sum", "reddit_sent_n"))
LOOKBACK_DAYS = 45


def _attention_counts(store: PITStore, symbol: str, as_of: date,
                      lookback: int = LOOKBACK_DAYS) -> pd.Series:
    """Daily total mention count for one coin (news + reddit), knowledge_time-gated."""
    start = as_of - timedelta(days=lookback)
    frames = [store.alt_signals_window([symbol], c, start, as_of) for c in ATTENTION_CONCEPTS]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.Series(dtype=float)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.groupby("dt")["value"].sum().sort_index()


class CryptoSentimentSignals:
    """Computes all three signals from the store. Read-only; pure inputs to signals.py."""

    def __init__(self, store: PITStore):
        self.store = store

    # -- Signal 1 --------------------------------------------------------
    def vol_forecast(self, symbol: str, as_of: date, trailing_vol: float) -> float:
        counts = _attention_counts(self.store, symbol, as_of)
        z = sg.attention_z(counts, as_of=as_of)
        return sg.vol_forecast(trailing_vol, z)

    # -- Signal 2 --------------------------------------------------------
    def equity_risk_appetite_level(self, as_of: date) -> float:
        """Market-wide equity sentiment on `as_of`: per-name (sent_sum/sent_n) averaged
        across all S&P names that scored that day, news and reddit pooled."""
        start = as_of - timedelta(days=3)
        syms = list(self.store.universe_as_of(as_of).symbol)
        if not syms:
            return float("nan")
        per_name: dict[str, list[float]] = {}
        for sum_c, n_c in SENT_CONCEPTS:
            sums = self.store.alt_signals_window(syms, sum_c, start, as_of)
            ns = self.store.alt_signals_window(syms, n_c, start, as_of)
            if sums.empty or ns.empty:
                continue
            # alt_signals_window gates by knowledge_time <= as_of (the leak guard), not by
            # dt <= as_of. In production knowledge_time ~= dt, so the two coincide -- but do
            # not depend on that: trim to dt <= as_of so a row dated after as_of can never
            # feed the level, whatever its knowledge_time.
            cutoff = pd.Timestamp(as_of)
            keep = lambda d: d[pd.to_datetime(d["dt"]) <= cutoff]  # noqa: E731
            sums, ns = keep(sums), keep(ns)
            if sums.empty or ns.empty:
                continue
            latest = lambda d: (d.sort_values("dt").groupby("symbol")["value"].last())  # noqa: E731
            s, n = latest(sums), latest(ns)
            for sym in s.index:
                if n.get(sym, 0) > 0:
                    per_name.setdefault(sym, []).append(s[sym] / n[sym])
        if not per_name:
            return float("nan")
        return sg.equity_risk_appetite(pd.Series({k: np.mean(v) for k, v in per_name.items()}))

    def equity_risk_appetite_change(self, as_of: date, window: int = 5) -> float:
        # Index by Timestamp (the pure function compares index <= Timestamp), but query the
        # store per calendar day with .date(). Each level is PIT (knowledge_time <= that day).
        dates = pd.date_range(as_of - timedelta(days=window + 3), as_of, freq="D")
        hist = pd.Series({ts: self.equity_risk_appetite_level(ts.date()) for ts in dates})
        return sg.equity_risk_appetite_change(hist, as_of=as_of, window=window)

    # -- Signal 3 --------------------------------------------------------
    def attention_spike(self, symbol: str, as_of: date) -> dict:
        return sg.attention_spike(_attention_counts(self.store, symbol, as_of), as_of=as_of)

    # -- everything, for logging (gated) ---------------------------------
    def compute_all(self, symbols: list[str], as_of: date,
                    trailing_vols: dict[str, float]) -> dict:
        """The full gated snapshot written to the rebalance record. Affects nothing."""
        era_level = self.equity_risk_appetite_level(as_of)
        era_change = self.equity_risk_appetite_change(as_of)
        per_coin = {}
        for s in symbols:
            spike = self.attention_spike(s, as_of)
            per_coin[s] = {
                "attention_z": None if not np.isfinite(spike["attention_z"]) else round(spike["attention_z"], 3),
                "spike": spike["spike"],
                "vol_forecast": _r(self.vol_forecast(s, as_of, trailing_vols.get(s, float("nan")))),
                "trailing_vol": _r(trailing_vols.get(s, float("nan"))),
            }
        return {
            "as_of": str(as_of),
            "equity_risk_appetite": _r(era_level),
            "equity_risk_appetite_change": _r(era_change),
            "per_coin": per_coin,
            "gated": True,   # NONE of this touches sizing or execution yet
        }


def _r(x, nd: int = 4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)
