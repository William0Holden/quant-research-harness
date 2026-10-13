"""Position sizing: ranker output -> long-only target weights.

Pure / testable. The two core ideas:

  * inverse-volatility weighting (risk parity-lite) scaled to a target portfolio vol,
  * fractional-Kelly gross scaling sized off HALF the estimated edge (a deliberate
    margin of safety against edge over-estimation — see ``quarter_kelly_scale``).

Nothing here touches the network or any external key; vol is estimated purely from the
PIT price store via ``prices_as_of``.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..pit.store import PITStore

# Assumed average pairwise correlation among the long book. Real long-only equity
# baskets are positively correlated (shared market beta), so naive sum-of-variances
# badly understates portfolio vol. We use a single flat rho as a transparent, robust
# proxy rather than estimating a full (noisy, n>>history) covariance matrix.
DEFAULT_AVG_CORR = 0.30
TRADING_DAYS = 252

# Conviction tilt bounds (see conviction_tilt). cap=2.0 means the strongest name can be
# at most 8x the weight of a floor-scoring one (2.0 / 0.25) before inverse-vol adjusts it.
DEFAULT_CONVICTION_CAP = 2.0
DEFAULT_CONVICTION_FLOOR = 0.25


def _daily_vol(store: PITStore, symbol: str, as_of: date, lookback: int) -> float:
    """Trailing daily-return stdev for one symbol, or NaN if too little history."""
    # Pull a touch more calendar history than `lookback` trading rows we need.
    px = store.prices_as_of([symbol], as_of, lookback_days=int(lookback * 2 + 20))
    if px.empty:
        return float("nan")
    s = px.sort_values("dt")["adj_close"].astype(float)
    s = s[s > 0]
    rets = np.log(s / s.shift(1)).dropna()
    rets = rets.tail(lookback)
    if len(rets) < 20:  # not enough to trust a stdev
        return float("nan")
    return float(rets.std(ddof=1))


def estimate_vols(
    symbols: list[str], store: PITStore, as_of: date, lookback: int = 126
) -> pd.Series:
    """Annualized vol per symbol (NaN where history is insufficient).

    A store MAY advertise `annualized_vols` to compute the whole book in one shot; the
    backtest's PanelStore does, because the default path below is one query per symbol
    per rebalance and a multi-year run makes tens of thousands of them. PITStore does
    not, and takes the loop. Both must produce the same estimator -- see
    tests/test_panel_store.py, which pins them together against the same prices.
    """
    fast = getattr(store, "annualized_vols", None)
    if fast is not None:
        return fast(symbols, as_of, lookback)

    out = {}
    for sym in symbols:
        dv = _daily_vol(store, sym, as_of, lookback)
        out[sym] = dv * np.sqrt(TRADING_DAYS) if np.isfinite(dv) else np.nan
    return pd.Series(out, dtype=float)


def portfolio_vol(weights: pd.Series, vols: pd.Series, avg_corr: float = DEFAULT_AVG_CORR) -> float:
    """Estimated portfolio vol under a flat-correlation assumption.

    var = sum_i w_i^2 s_i^2 + rho * sum_{i!=j} w_i w_j s_i s_j
    Aligns weights/vols on their shared index and drops NaNs.
    """
    df = pd.concat([weights.rename("w"), vols.rename("s")], axis=1).dropna()
    if df.empty:
        return 0.0
    w = df["w"].to_numpy()
    s = df["s"].to_numpy()
    contrib = w * s
    diag = float(np.sum(contrib**2))
    cross = float(contrib.sum() ** 2 - diag)  # sum_{i!=j} (w_i s_i)(w_j s_j)
    var = diag + avg_corr * cross
    return float(np.sqrt(max(var, 0.0)))


def conviction_tilt(
    scores: pd.Series, cap: float = DEFAULT_CONVICTION_CAP,
    floor: float = DEFAULT_CONVICTION_FLOOR,
) -> pd.Series:
    """Composite z-score -> a bounded, strictly-positive relative-conviction multiplier.

    Without this, every name in the top bucket is sized identically: the sizer knows the
    strongest name scored 2.4 and the marginal one scored 0.3, and then discards that.

    Saturating (clip at ``cap``) is what keeps this from becoming a concentration bet on
    whichever name has the most extreme composite z -- extreme z-scores are usually a
    thin-data artefact, not extra edge. Flooring at ``floor`` keeps a barely-included
    name at a small but non-zero weight rather than silently dropping it, so the bucket's
    membership decision stays with the ranker, not with the sizer.
    """
    s = scores.astype(float).clip(lower=floor, upper=cap)
    return s.fillna(floor)


def vol_target_weights(
    symbols: list[str],
    store: PITStore,
    as_of: date,
    target_vol: float = 0.10,
    lookback: int = 126,
    avg_corr: float = DEFAULT_AVG_CORR,
    tilt: pd.Series | None = None,
) -> pd.Series:
    """Inverse-vol weights (sum=1) scaled so estimated portfolio vol ~= ``target_vol``.

    Returns a Series indexed by the input symbols. Names with insufficient price
    history get zero weight and are dropped. The returned weights sum to <= 1
    (== the gross scale needed to hit target vol, capped at 1.0 — we never lever
    the long book in v1).

    ``tilt`` (optional) multiplies the inverse-vol weights before normalisation, so a
    higher-conviction name gets a bigger share of the same risk budget. It changes the
    RELATIVE weights only: the book is still scaled to ``target_vol`` afterwards, so
    tilting cannot quietly raise portfolio risk.
    """
    vols = estimate_vols(symbols, store, as_of, lookback)
    vols = vols[vols.notna() & (vols > 0)]
    if vols.empty:
        return pd.Series(dtype=float)

    inv = 1.0 / vols
    if tilt is not None:
        t = tilt.reindex(inv.index).astype(float)
        # A name with no score keeps its untilted inverse-vol weight rather than being
        # dropped: missing conviction is not negative conviction.
        inv = inv * t.fillna(1.0).clip(lower=0.0)
        inv = inv[inv > 0]
        if inv.empty:
            return pd.Series(dtype=float)
        vols = vols.reindex(inv.index)
    w = inv / inv.sum()  # normalized (optionally tilted) inverse-vol, sums to 1

    # Scale the whole book so its estimated vol matches the target. base_vol is the
    # vol of the fully-invested (sum=1) inverse-vol book; the scalar that hits target
    # is target/base. Cap at 1.0 (long-only, unlevered).
    base_vol = portfolio_vol(w, vols, avg_corr)
    scale = 1.0 if base_vol <= 0 else min(target_vol / base_vol, 1.0)
    return w * scale


def quarter_kelly_scale(expected_edge: float, variance: float) -> float:
    """Fractional-Kelly gross-exposure fraction in [0, 1].

    Kelly optimal fraction for a continuous bet ~ edge / variance. We take a QUARTER
    of that (0.25 *) for robustness, AND we size off only HALF the estimated edge
    (``edge/2``) as a second, deliberate haircut against edge over-estimation —
    parameter risk is the dominant risk in live sizing.

        f = 0.25 * ((edge / 2) / variance)

    Clipped to [0, 1]. Non-positive edge or non-positive variance -> 0.
    """
    if variance is None or variance <= 0 or expected_edge is None or expected_edge <= 0:
        return 0.0
    f = 0.25 * ((expected_edge / 2.0) / variance)
    return float(np.clip(f, 0.0, 1.0))


def hysteresis_members(
    ranked_df: pd.DataFrame, top_syms: list[str], prev_w: dict[str, float] | None,
    hold_bucket: int | None,
) -> list[str]:
    """Membership with a retention band: enter only from the top bucket, but a name we
    ALREADY HOLD stays while it is still ranked at or above `hold_bucket`.

    Why: with ~100 names the top-quintile boundary is crossed by dozens of names a week
    on rank noise alone (the live book swapped 11-41 of 101 names every week, Jul-Aug
    2026, and paid for it: 780 orders, $317k traded on a $57k book). A two-sided
    threshold (Schmitt trigger) is the standard remedy: a name must fall out of the top
    `hold_bucket` before it is sold, not merely slip out of the top quintile.

    hold_bucket=None (or 0) disables this and returns top_syms unchanged."""
    if not hold_bucket or not prev_w:
        return list(top_syms)
    keep = set(top_syms)
    b = ranked_df["bucket"]
    for sym, w in prev_w.items():
        if w > 0 and sym in b.index and sym not in keep:
            bk = b.at[sym]
            if bk == bk and bk >= hold_bucket:
                keep.add(sym)
    # preserve ranked order for determinism
    return [s for s in ranked_df.index if s in keep]


def build_long_only(
    ranked_df: pd.DataFrame,
    store: PITStore,
    as_of: date,
    cfg: dict | None = None,
    prev_w: dict[str, float] | None = None,
) -> dict[str, float]:
    """Top-bucket -> vol-targeted, Kelly-scaled long-only weights.

    cfg keys (all optional):
        target_vol  (default 0.10)
        lookback    (default 126)
        avg_corr    (default DEFAULT_AVG_CORR)
        conviction  (default False) -> size by composite score, not just inverse vol
        conviction_cap / conviction_floor -> tilt bounds (see conviction_tilt)
        expected_edge, variance  -> if both given, apply quarter_kelly_scale
        kelly_scale (default 1.0) -> explicit gross multiplier when edge/var absent
        hold_bucket (default None) -> retain held names ranked >= this bucket
                                      (see hysteresis_members); needs prev_w
    prev_w: symbol -> weight currently held (only used by hold_bucket).
    """
    cfg = cfg or {}
    if ranked_df is None or ranked_df.empty or "bucket" not in ranked_df:
        return {}
    buckets = ranked_df["bucket"].dropna()
    if buckets.empty:
        return {}
    top = buckets.max()
    top_syms = list(ranked_df.index[ranked_df["bucket"] == top])
    if not top_syms:
        return {}
    top_syms = hysteresis_members(ranked_df, top_syms, prev_w, cfg.get("hold_bucket"))

    tilt = None
    if cfg.get("conviction", False) and "score" in ranked_df:
        tilt = conviction_tilt(
            ranked_df.loc[top_syms, "score"],
            cap=cfg.get("conviction_cap", DEFAULT_CONVICTION_CAP),
            floor=cfg.get("conviction_floor", DEFAULT_CONVICTION_FLOOR),
        )

    w = vol_target_weights(
        top_syms,
        store,
        as_of,
        target_vol=cfg.get("target_vol", 0.10),
        lookback=cfg.get("lookback", 126),
        avg_corr=cfg.get("avg_corr", DEFAULT_AVG_CORR),
        tilt=tilt,
    )
    if w.empty:
        return {}

    if "expected_edge" in cfg and "variance" in cfg:
        gross = quarter_kelly_scale(cfg["expected_edge"], cfg["variance"])
    else:
        gross = float(cfg.get("kelly_scale", 1.0))
    w = w * gross
    return {s: float(v) for s, v in w.items() if v > 0}
