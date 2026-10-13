"""Trend exposure -> the weights we actually hold. Vol target, gross cap, sentiment overlay.

THE VOL TARGET IS A RISK DIAL, NOT AN ALPHA LEVER. MEASURED, ON THIS BOOK:

    vol target   avg gross    CAGR   Sharpe   maxDD   Calmar
       10%          0.10       2.6%    0.42   -10.7%   0.24
       20%          0.20       5.0%    0.44   -19.8%   0.26
       40%          0.38      10.3%    0.52   -35.4%   0.29
      none          0.51      13.5%    0.54   -49.0%   0.28
    BTC B&H         1.00      15.9%    0.54   -76.7%   0.21

Sharpe is FLAT (0.42 -> 0.54). Vol targeting buys no risk-adjusted return whatsoever -- it
only decides how much of the book is invested. Drawdown falls because you are holding less,
which is not a strategy. Anyone reporting the -10.7% drawdown of the 10% target as a result
would be reporting "I was 90% in cash."

So the target is chosen by RISK APPETITE, and 40% is picked because the realized book vol
then lands near 25% (trend exposure averages ~0.5, which multiplies the vol scalar DOWN --
so a "40% target" is not a 40% book), giving a kill-switch ratio of 50/25 = 2.0x, in line
with the equity book's 25/10 = 2.5x. Observed maxDD at this setting is -35%, comfortably
inside the 50% switch, so the switch stays a genuine tail breaker rather than a certainty.

Crypto's leverage effect is also partially INVERTED (vol rises with POSITIVE returns; melt-ups
are as violent as melt-downs), which is why vol targeting cannot do here what it does in
equities. What it DOES buy is time-underwater: 2.96 years -> 1.50. That is what you sit through.

GROSS IS HARD-CAPPED AT 1.0. A long-only coin book is never levered, whatever the vol
target asks for in a calm regime. Crypto daily excess kurtosis runs 8-49 (vs ~3-6 for
equity indices); levering into that is how people die.

THE SENTIMENT OVERLAY IS A RISK SIGNAL, NOT A DIRECTIONAL ONE.
The literature is consistent that social sentiment predicts VOLATILITY and UNCERTAINTY, not
returns (Antweiler & Frank JF 2004; "The Extremity Premium" arXiv 2602.07018 finds extreme
greed adds +5.5pp and extreme fear +3.9pp of uncertainty, while the contrarian DIRECTIONAL
strategy was NOT significant, t=1.02). Sentiment extremes are also exactly when market
makers widen and step away -- so fills degrade precisely when the signal fires.

Therefore extremity SCALES GROSS DOWN. It never picks direction, and it can never scale up.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

VOL_TARGET = 0.40          # see below -- realized lands ~25%, NOT 40%
MAX_GROSS = 1.0            # never lever a long-only coin book
VOL_LOOKBACK = 30
MIN_VOL_OBS = 20
ANNUALIZE = 365            # crypto trades every day; there is no 252-day year here

# Sentiment overlay: m = 1 - OVERLAY_K * extremity, floored. Never exceeds 1.0.
OVERLAY_K = 0.5
OVERLAY_FLOOR = 0.5        # the overlay may halve the book; it may never flatten or lever it


def realized_vol(close: pd.Series, lookback: int = VOL_LOOKBACK) -> float:
    """Annualized realized vol from the last `lookback` simple returns. NaN if too thin.

    NaN, never 0.0: a zero vol makes the 1/vol scalar infinite and would hand the book
    unbounded leverage. The caller must treat NaN as "do not size", not as "no risk".
    """
    c = pd.Series(close).astype(float)
    r = c.pct_change().dropna().tail(lookback)
    if len(r) < MIN_VOL_OBS:
        return float("nan")
    sd = float(r.std())
    return sd * math.sqrt(ANNUALIZE) if sd > 0 else float("nan")


def vol_scalar(vol: float, vol_target: float = VOL_TARGET,
               max_gross: float = MAX_GROSS) -> float:
    """target/realized, clipped to [0, max_gross]. NaN vol -> 0.0 (refuse to size)."""
    if vol is None or not np.isfinite(vol) or vol <= 0:
        return 0.0
    return float(min(vol_target / vol, max_gross))


def extremity(z: float) -> float:
    """|sentiment z-score| mapped to [0, 1]. NaN (GPU down / too thin) -> 0.0 = no effect.

    Saturates at 2 sigma: beyond that the score is far more likely to be a thin-data
    artefact or a bot swarm than information. 14-56% of crypto social accounts are bots or
    suspended, and 93% of bot-shared invite links point at pump-and-dump channels -- an
    extreme reading is exactly what a coordinated pump produces.
    """
    if z is None or not np.isfinite(z):
        return 0.0
    return float(min(abs(z) / 2.0, 1.0))


def overlay_multiplier(sent_z: float | None, k: float = OVERLAY_K,
                       floor: float = OVERLAY_FLOOR) -> float:
    """Sentiment EXTREMITY -> a gross multiplier in [floor, 1.0]. Never > 1.0.

    NaN/None sentiment -> 1.0. This is the GPU-down path and it is load-bearing: GPU worker is
    shared with the user, so when it is unavailable the sentiment score goes stale, the
    provider returns NaN, and the book must fall back to trading on trend ALONE rather than
    erroring or blocking. Degrade to the price-only strategy, never to a crash.
    """
    m = 1.0 - k * extremity(sent_z)
    return float(max(min(m, 1.0), floor))


def build_crypto_targets(
    trend_exposure: dict[str, float],
    closes: dict[str, pd.Series],
    sent_z: float | None = None,
    vol_target: float = VOL_TARGET,
    max_gross: float = MAX_GROSS,
    overlay_on: bool = False,
) -> tuple[dict[str, float], dict]:
    """The whole sizing pipeline, pure.

    trend_exposure: symbol -> fraction of rules long, in [0, 1] (NaN = not warmed up).
    closes:         symbol -> price history (for the vol estimate).
    sent_z:         aggregate crypto sentiment z-score, or None. Only used if overlay_on.

    Returns (weights, diagnostics). Weights are equal-notional across the universe, each
    leg scaled by its own trend exposure and its own vol scalar, then the whole book scaled
    by the overlay multiplier and finally hard-capped at max_gross.

    `m` is a SCALAR multiplier, which is what makes the overlay exactly attributable:
        r_book(t) = m(t-1) * r_trend(t)
    so `trend_weights` (the overlay-off counterfactual) is recorded alongside `weights`,
    and the shadow overlay-off NAV can be reconstructed exactly. If the overlay ever makes
    or loses money, we will know it was the overlay.
    """
    syms = sorted(trend_exposure)
    n = len(syms)
    per_leg = 1.0 / n if n else 0.0

    trend_w: dict[str, float] = {}
    vols: dict[str, float] = {}
    for s in syms:
        e = trend_exposure.get(s, float("nan"))
        v = realized_vol(closes[s]) if s in closes else float("nan")
        vols[s] = v
        if e is None or not np.isfinite(e):
            continue                      # not warmed up: hold nothing, do not guess
        w = per_leg * float(e) * vol_scalar(v, vol_target, max_gross)
        if w > 0:
            trend_w[s] = w

    # Gross cap applies to the TREND book (the overlay can only ever reduce from here).
    g = sum(trend_w.values())
    if g > max_gross:
        trend_w = {s: w * (max_gross / g) for s, w in trend_w.items()}
        g = max_gross

    m = overlay_multiplier(sent_z) if overlay_on else 1.0
    weights = {s: w * m for s, w in trend_w.items()}

    diag = {
        "trend_exposure": {s: (None if not np.isfinite(trend_exposure.get(s, float("nan")))
                               else float(trend_exposure[s])) for s in syms},
        "realized_vol": {s: (None if not np.isfinite(vols[s]) else round(vols[s], 4))
                         for s in syms},
        "vol_target": vol_target,
        "trend_gross": round(g, 6),                # overlay-OFF gross (counterfactual)
        "overlay_on": overlay_on,
        "overlay_m": round(m, 6),                  # log every run: enables exact attribution
        "sent_z": None if sent_z is None or not np.isfinite(sent_z) else round(float(sent_z), 4),
        "gross": round(sum(weights.values()), 6),
        "n_held": len(weights),
    }
    return weights, diag
