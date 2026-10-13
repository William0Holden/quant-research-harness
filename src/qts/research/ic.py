"""Per-provider information-coefficient analysis + trailing IC weights.

IC(t) = cross-sectional Spearman correlation between a provider's sign-adjusted
z-score at decision date t (PIT: knowledge_time <= t) and the forward return
over (next bar after t] -> (next bar after t+1] — the exact window the backtest
trades. Future prices score already-decided ranks only; they never feed a rank.

Each row is indexed by its forward window's EXIT bar (not the decision date t):
a row only becomes usable once that window has fully resolved, so trailing_ic_
weights' `index < as_of` filter never admits a row whose forward return still
reaches past as_of (one-bar look-ahead guard).
"""
from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from ..backtest.engine import _adj_panel, _next_bar, _trading_days, make_schedule
from ..features.base import FeatureProvider
from ..features.ranker import compute_features, cross_section_zscore
from ..pit.store import PITStore

MIN_NAMES = 5


def provider_ic(
    store: PITStore, providers: list[FeatureProvider],
    start: date, end: date, freq: str = "W-FRI",
) -> pd.DataFrame:
    """Per-provider Spearman IC per rebalance period.

    index = forward-window exit bar (NOT the decision date t) — each row is stamped
    with the date its forward return actually resolves, so a leak-safe `index < as_of`
    filter (see trailing_ic_weights) never admits a row whose window reaches past as_of.
    """
    days = _trading_days(store)
    panel = _adj_panel(store)
    schedule = make_schedule(days, start, end, freq)
    rows, idx = [], []
    for i in range(len(schedule) - 1):
        t, t_next = pd.Timestamp(schedule[i]), pd.Timestamp(schedule[i + 1])
        entry, exit_ = _next_bar(panel.index, t), _next_bar(panel.index, t_next)
        if entry is None or exit_ is None:
            continue
        syms = [s for s in store.universe_as_of(t.date())["symbol"]
                if s in panel.columns]
        if len(syms) < MIN_NAMES:
            continue
        fwd = (panel.loc[exit_, syms] / panel.loc[entry, syms] - 1.0)
        feats = compute_features(store, providers, syms, t.date())
        zdf = cross_section_zscore(feats, providers)
        row = {}
        for col in zdf.columns:
            pair = pd.concat([zdf[col], fwd], axis=1, keys=["z", "r"]).dropna()
            if len(pair) < MIN_NAMES:
                row[col] = np.nan
            elif pair["z"].nunique() <= 1 or pair["r"].nunique() <= 1:
                row[col] = np.nan
            else:
                row[col] = pair["z"].corr(pair["r"], method="spearman")
        rows.append(row)
        idx.append(exit_)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in ic_df.columns:
        s = ic_df[col].dropna()
        n = len(s)
        std = s.std()
        out[col] = {
            "mean_ic": s.mean() if n else np.nan,
            "std_ic": std if n else np.nan,
            "t_stat": (s.mean() / std * np.sqrt(n)) if n > 1 and std else np.nan,
            "hit_rate": (s > 0).mean() if n else np.nan,
            "n_periods": n,
        }
    return pd.DataFrame(out).T


def trailing_ic_weights(
    ic_df: pd.DataFrame, as_of: date, window: int = 26, floor: float = 0.0,
) -> dict[str, float] | None:
    """Past-only: uses IC rows whose forward window's exit bar is STRICTLY BEFORE as_of
    (leak-safe by construction — a row is only eligible once its window has resolved)."""
    past = ic_df[ic_df.index < pd.Timestamp(as_of)].tail(window)
    if past.empty:
        return None
    mean = past.mean().fillna(0.0).clip(lower=floor)
    total = mean.sum()
    if not total or np.isnan(total):
        return None
    return (mean / total).to_dict()


def make_weights_fn(
    ic_df: pd.DataFrame, window: int = 26,
) -> Callable[[date], dict | None]:
    def fn(as_of: date) -> dict | None:
        return trailing_ic_weights(ic_df, as_of, window=window)
    return fn


SHORT_WINDOW = 13   # ~one quarter of weekly ICs — the fast horizon
LONG_WINDOW = 52    # ~one year — the slow horizon
MIN_IC_ROWS = 8     # below this a provider rides its equal-weight prior only
SHRINK = 0.5        # 50% toward equal weight: adapt without whipsawing on noise


def blended_ic_weights(
    ic_df: pd.DataFrame, as_of: date, live_names: list[str],
    short_window: int = SHORT_WINDOW, long_window: int = LONG_WINDOW,
    shrink: float = SHRINK, min_rows: int = MIN_IC_ROWS,
) -> tuple[dict[str, float] | None, dict]:
    """Blend short- and long-horizon trailing mean IC into live composite weights.

    Leak-safe by construction: ic_df rows are indexed by their forward-window
    EXIT bar (see provider_ic), so `index < as_of` only admits fully-resolved
    periods. Negative blended IC is floored at 0 (a bad signal is silenced, never
    traded inverted — sign flips are a deliberate human decision). The floored
    IC weights are then shrunk toward equal weight and L1-normalized."""
    past = ic_df[ic_df.index < pd.Timestamp(as_of)]
    diag: dict = {"n_rows_used": int(len(past)), "short_window": short_window,
                  "long_window": long_window, "shrink": shrink}
    raw: dict[str, float] = {}
    for name in live_names:
        s = past[name].dropna() if name in past.columns else pd.Series(dtype=float)
        if len(s) < min_rows:
            continue
        blended = 0.5 * s.tail(short_window).mean() + 0.5 * s.tail(long_window).mean()
        raw[name] = max(0.0, float(blended))
    if not raw:
        diag["reason"] = "insufficient_history"
        return None, diag
    ic_total = sum(raw.values())
    equal = 1.0 / len(live_names)
    w = {}
    for name in live_names:
        ic_part = (raw.get(name, 0.0) / ic_total) if ic_total > 0 else equal
        # a provider with too little history gets the equal prior on both halves
        if name not in raw:
            ic_part = equal
        w[name] = shrink * equal + (1.0 - shrink) * ic_part
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    diag["raw_blended_ic"] = {k: round(v, 5) for k, v in raw.items()}
    return w, diag
