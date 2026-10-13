"""Crypto daily bars from Alpaca. Free, unauthenticated, back to 2021-01-01.

PIT: knowledge_time = the bar's date. Crypto has no close, so "daily" is a convention --
Alpaca anchors at 00:00 UTC and so do we. That anchor is LOAD-BEARING and arbitrary: if a
strategy's Sharpe swings materially when the bar boundary moves, the edge is an artifact of
bar alignment, not a signal. `qts.crypto.backtest --anchor-sweep` runs that check.

PAGINATION IS A TRAP. The /bars endpoint caps at `limit` rows ACROSS ALL SYMBOLS, so a
multi-symbol request silently truncates and the later symbols come back empty -- which reads
exactly like "this coin has no history." Always follow next_page_token; never trust a
single-shot multi-symbol pull. (This bit me once already during research.)
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import requests

BARS = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
START = "2021-01-01"


def fetch_bars(symbol: str, start: str = START, timeframe: str = "1Day") -> pd.Series:
    """Close series for one pair, indexed by date. Paginated; empty Series if none."""
    out: list[dict] = []
    token = None
    while True:
        params = {"symbols": symbol, "timeframe": timeframe, "start": start, "limit": 10000}
        if token:
            params["page_token"] = token
        r = requests.get(BARS, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        out += j.get("bars", {}).get(symbol, [])
        token = j.get("next_page_token")
        if not token:
            break
    if not out:
        return pd.Series(dtype=float, name=symbol)
    s = pd.Series({pd.Timestamp(b["t"][:10]): float(b["c"]) for b in out}, name=symbol)
    return s[~s.index.duplicated()].sort_index()


def fetch_panel(symbols, start: str = START) -> pd.DataFrame:
    """symbol -> close panel. One paginated request per symbol (see the trap above)."""
    cols = [fetch_bars(s, start) for s in symbols]
    cols = [c for c in cols if not c.empty]
    return pd.concat(cols, axis=1).sort_index() if cols else pd.DataFrame()


def latest_quote(symbol: str) -> dict:
    """Live top-of-book. Returns bid, ask, mid, half_spread_bps.

    This is the arrival mid the cost ledger anchors on -- snapshot it at DECISION time or
    implementation shortfall is not computable after the fact.
    """
    r = requests.get("https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes",
                     params={"symbols": symbol}, timeout=20)
    r.raise_for_status()
    q = r.json()["quotes"][symbol]
    bid, ask = float(q["bp"]), float(q["ap"])
    mid = (bid + ask) / 2
    return {"bid": bid, "ask": ask, "mid": mid,
            "half_spread_bps": (ask - bid) / 2 / mid * 1e4 if mid else float("nan")}
