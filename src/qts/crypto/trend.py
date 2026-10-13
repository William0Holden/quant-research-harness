"""Trend ensemble: the backtestable core of the crypto book.

WHY AN ENSEMBLE AND NOT THE BEST RULE.
On 2021-2026 daily bars, every trend rule tested cut max drawdown roughly in half:

    BTC buy & hold      CAGR 15.9%  Sharpe 0.54  maxDD -76.7%
    BTC px > 100d MA    CAGR 17.5%  Sharpe 0.63  maxDD -39.8%
    ETH buy & hold      CAGR  0.5%  Sharpe 0.36  maxDD -79.3%
    ETH px > 200d MA    CAGR 24.3%  Sharpe 0.71  maxDD -39.2%

But the BEST rule differs between the two assets (100d on BTC, 200d on ETH), and the
Sharpe improvement is not statistically significant -- at 5.5 years the single-trial
significance threshold is Sharpe >= 0.84, which not even BTC's own 0.54 clears. A rule
that is best on one asset and not the other, by an insignificant margin, is noise.

The DIRECTION is robust (every rule, both assets, halved the drawdown). The SPECIFIC RULE
is not. So we never pick a winner: exposure is the FRACTION of rules currently long. The
book scales in and out instead of flipping, and no single lookback can be the thing that
was overfit.

Zero free parameters are fitted here. Given ~4 effective independent regime observations
in the sample (2021 bull / 2022 bear / 2023-24 recovery / 2025-26), that is the point.

PIT: every rule at time t uses only closes <= t. The caller trades on t+1.
"""
from __future__ import annotations

import pandas as pd

# Six rules, fixed, never tuned. Each maps a close series -> bool (long or flat).
RULES: tuple[str, ...] = (
    "px>50ma", "px>100ma", "px>200ma", "ma20>ma100", "ma50>ma200", "ret90>0",
)
# Longest lookback any rule needs. Before this many bars, exposure is undefined (NaN),
# never 0.0 -- "no opinion" and "flat on purpose" are different states and conflating
# them would silently backtest a cash position as a trend decision.
WARMUP = 200


def rule_signals(close: pd.Series) -> pd.DataFrame:
    """close (indexed by date) -> DataFrame[bool] of each rule's long/flat state.

    NaN (not False) until each rule has enough history.
    """
    c = pd.Series(close).astype(float)
    out = pd.DataFrame(index=c.index)
    out["px>50ma"] = c > c.rolling(50).mean()
    out["px>100ma"] = c > c.rolling(100).mean()
    out["px>200ma"] = c > c.rolling(200).mean()
    out["ma20>ma100"] = c.rolling(20).mean() > c.rolling(100).mean()
    out["ma50>ma200"] = c.rolling(50).mean() > c.rolling(200).mean()
    out["ret90>0"] = c.pct_change(90) > 0

    # A comparison against NaN yields False, which would masquerade as a real "flat"
    # decision during warmup. Mask it back to NaN so `exposure` can refuse to trade.
    warm = pd.DataFrame({
        "px>50ma": c.rolling(50).mean().notna(),
        "px>100ma": c.rolling(100).mean().notna(),
        "px>200ma": c.rolling(200).mean().notna(),
        "ma20>ma100": c.rolling(100).mean().notna(),
        "ma50>ma200": c.rolling(200).mean().notna(),
        "ret90>0": c.pct_change(90).notna(),
    })
    return out.where(warm)


def exposure(close: pd.Series) -> pd.Series:
    """Fraction of rules currently long, in [0, 1]. NaN until every rule has warmed up.

    This is the raw trend exposure BEFORE vol targeting. It is not a weight yet.
    """
    sig = rule_signals(close)
    frac = sig.mean(axis=1)                 # mean of bools = fraction long
    return frac.where(sig.notna().all(axis=1))


def exposure_as_of(close: pd.Series, as_of) -> float:
    """PIT scalar: trend exposure using only closes <= as_of. NaN if not warmed up.

    The live path calls this; the backtest calls `exposure` vectorised. Both must agree,
    which `tests/test_trend_ensemble.py` pins.
    """
    ts = pd.Timestamp(as_of)
    c = pd.Series(close).astype(float)
    c = c[c.index <= ts]
    if len(c) < WARMUP:
        return float("nan")
    e = exposure(c)
    return float(e.iloc[-1]) if len(e) else float("nan")
