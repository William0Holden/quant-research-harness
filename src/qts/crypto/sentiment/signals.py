"""The three crypto sentiment RISK signals, as pure functions.

Pure = no store, no network, no clock. Each takes plain pandas data and returns a float (or
NaN when the data is too thin to mean anything -- "no opinion", never a fabricated zero). The
store-reading wrappers live in providers.py; the risk-IC measurement lives in research.py.

Every signal targets RISK (volatility, drawdown), not return. That is the whole thesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Standardization windows for attention. "recent" = the last few days of mention volume;
# "baseline" = the trailing distribution it is judged against.
RECENT_DAYS = 3
BASELINE_DAYS = 30
MIN_BASELINE_OBS = 10

# Signal 1: how strongly attention lifts the vol forecast. Bounded so a mention frenzy can
# raise the forecast but never invert or explode it. Chosen conservatively; the measurement
# harness re-tunes it only if the risk-IC record earns it.
VOL_FORECAST_K = 0.25
ATTENTION_Z_CLIP = 3.0

# Signal 3: attention-spike threshold in standard deviations.
SPIKE_Z = 2.0


def attention_z(counts: pd.Series, as_of=None,
                recent_days: int = RECENT_DAYS, baseline_days: int = BASELINE_DAYS) -> float:
    """Standardized recent mention volume for one coin. NaN when the baseline is too thin.

    counts: daily mention count (news+reddit) indexed by date. as_of: only data <= as_of is
    used (PIT); None = use everything. Returns (recent_mean - baseline_mean) / baseline_std.
    """
    s = pd.Series(counts).dropna().sort_index()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) < MIN_BASELINE_OBS + 1:
        return float("nan")
    recent = s.iloc[-recent_days:].mean()
    baseline = s.iloc[-(baseline_days + recent_days):-recent_days]
    if len(baseline) < MIN_BASELINE_OBS:
        return float("nan")
    # Mention counts are COUNT data, so a low-activity or perfectly flat baseline has ~zero
    # sample std -- which would make any single mention look like an enormous surge (or, for
    # a constant series, blow up to NaN). Floor the std at a Poisson scale (sqrt of the mean
    # level): for count data the natural noise is ~sqrt(mean), so this regularizes sparse
    # coins and treats a genuinely steady tape as a zero anomaly rather than an undefined one.
    sd = max(float(baseline.std(ddof=0)), np.sqrt(max(float(baseline.mean()), 1.0)))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float((recent - baseline.mean()) / sd)


# ---- Signal 1: sentiment-augmented volatility forecast -------------------------------
def vol_forecast(trailing_vol: float, att_z: float,
                 k: float = VOL_FORECAST_K, clip: float = ATTENTION_Z_CLIP) -> float:
    """Lift the trailing realized-vol estimate by recent attention.

        vol_forecast = trailing_vol * (1 + k * clip(attention_z))

    High attention -> higher forecast -> the vol-targeted sizer de-risks BEFORE the move.
    Attention below its baseline (negative z) trims the forecast slightly. Never negative:
    the multiplier is floored so a quiet tape cannot drive the forecast to zero (which would
    make the sizer lever up). NaN attention -> fall back to trailing_vol unchanged (the
    GPU-down / thin-data path)."""
    if trailing_vol is None or not np.isfinite(trailing_vol) or trailing_vol <= 0:
        return float("nan")
    if att_z is None or not np.isfinite(att_z):
        return float(trailing_vol)                       # no opinion -> don't touch the estimate
    z = float(np.clip(att_z, -clip, clip))
    mult = max(1.0 + k * z, 0.25)                        # floor: never collapse the forecast
    return float(trailing_vol * mult)


# ---- Signal 2: cross-asset equity risk appetite --------------------------------------
def equity_risk_appetite(equity_sent: pd.Series) -> float:
    """Aggregate equity-market sentiment into one risk-appetite scalar (mean across names).

    equity_sent: per-symbol decayed sentiment (one value per S&P name) for a given day. The
    market-level mean is the risk-appetite level; NaNs (thin names) are ignored, not zeroed.
    NaN when nothing scored. This is the LEVEL; `equity_risk_appetite_change` gives the move
    that the signal actually trades on."""
    s = pd.Series(equity_sent).dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())


def equity_risk_appetite_change(level_history: pd.Series, as_of=None, window: int = 5) -> float:
    """Short-horizon change in equity risk appetite. A sharp DROP is the risk signal:
    equity risk-off leads crypto drawdown (crypto-equity corr ~0.4, 0.7+ in the tail).

    level_history: daily risk-appetite level (from equity_risk_appetite). Returns
    level(as_of) - level(as_of - window). NaN when history is too short."""
    s = pd.Series(level_history).dropna().sort_index()
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) <= window:
        return float("nan")
    return float(s.iloc[-1] - s.iloc[-1 - window])


# ---- Signal 3: attention spike / novelty ---------------------------------------------
def attention_spike(counts: pd.Series, as_of=None, threshold: float = SPIKE_Z) -> dict:
    """Detect a surge in mention volume vs the trailing baseline. Returns both the continuous
    z and a boolean flag, so callers can use either. A spike flags discrete events (hacks,
    depegs, ETF decisions) where being hours faster than a daily rebalance matters for
    getting OUT. NaN z -> not a spike (thin data is not an event)."""
    z = attention_z(counts, as_of=as_of)
    spike = bool(np.isfinite(z) and z >= threshold)
    return {"attention_z": z, "spike": spike, "threshold": threshold}


# ---- risk targets (what the signals are graded against) ------------------------------
def forward_realized_vol(prices: pd.Series, as_of, horizon: int = 10, ann: int = 365) -> float:
    """Annualized realized vol over the `horizon` days AFTER as_of. The grading target for
    signals 1 and 3. NaN if the future window is not fully present (no partial-window leak)."""
    p = pd.Series(prices).sort_index()
    fut = p[p.index > pd.Timestamp(as_of)].iloc[:horizon]
    if len(fut) < horizon:
        return float("nan")
    r = fut.pct_change().dropna()
    return float(r.std() * np.sqrt(ann)) if len(r) > 1 and r.std() > 0 else float("nan")


def forward_drawdown(prices: pd.Series, as_of, horizon: int = 10) -> float:
    """Worst peak-to-trough drawdown over the `horizon` days after as_of (a NEGATIVE number).
    The grading target for signal 2. NaN if the window is not fully present."""
    p = pd.Series(prices).sort_index()
    fut = p[p.index > pd.Timestamp(as_of)].iloc[:horizon]
    if len(fut) < horizon:
        return float("nan")
    nav = fut / fut.iloc[0]
    return float((nav / nav.cummax() - 1).min())
