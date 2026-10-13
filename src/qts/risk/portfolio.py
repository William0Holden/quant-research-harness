"""Compose sizing + regime gross-scaling + limits into a final target portfolio."""
from __future__ import annotations

from datetime import date

import pandas as pd

from .limits import apply_limits
from .regime import DEGRADED_GROSS_MULTIPLIER, regime_state
from .sizing import build_long_only


def build_target_portfolio(
    ranked_df: pd.DataFrame,
    store,
    as_of: date,
    equity: float,
    regime_on: bool = True,
    cfg: dict | None = None,
    regime_series: dict | None = None,
    prev_w: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict]:
    """Full sizing/risk pipeline -> (weights, diagnostics).

    Steps:
      1. sizing.build_long_only: top bucket -> vol-targeted, Kelly-scaled weights,
      2. regime overlay (optional): multiply gross by regime_state.gross_multiplier,
      3. limits.apply_limits: per-position cap + optional name cap + renormalize.

    Returns long-only weights summing to <= 1 (remainder is cash) and a small
    diagnostics dict: gross, n_positions, regime_state, kelly_scale, equity.

    cfg keys (optional): everything build_long_only accepts, plus
        max_position (default 0.10), max_names (default None).
    prev_w: currently held weights, for cfg["hold_bucket"] membership hysteresis.
    """
    cfg = cfg or {}
    diagnostics_regime = None

    base = build_long_only(ranked_df, store, as_of, cfg, prev_w=prev_w)

    # 2. regime gross-scaling
    gross_mult = 1.0
    if regime_on:
        try:
            diagnostics_regime = regime_state(as_of, series=regime_series)
            gross_mult = float(diagnostics_regime.get("gross_multiplier", 1.0))
        except Exception as e:  # noqa: BLE001  never let a macro fetch sink sizing
            # Fail CLOSED. regime_state already absorbs per-series failures and caps
            # gross itself; reaching this handler means the brake is wholly unavailable,
            # which is the least safe moment to size at full exposure.
            gross_mult = DEGRADED_GROSS_MULTIPLIER
            diagnostics_regime = {
                "error": f"regime evaluation failed: {type(e).__name__}: {e}",
                "degraded": True,
                "gross_multiplier": gross_mult,
            }
    base = {s: w * gross_mult for s, w in base.items()}

    # 3. hard limits
    weights = apply_limits(
        base,
        max_position=cfg.get("max_position", 0.10),
        max_names=cfg.get("max_names", None),
    )

    gross = float(sum(weights.values()))
    if "expected_edge" in cfg and "variance" in cfg:
        from .sizing import quarter_kelly_scale

        kelly_scale = quarter_kelly_scale(cfg["expected_edge"], cfg["variance"])
    else:
        kelly_scale = float(cfg.get("kelly_scale", 1.0))

    diagnostics = {
        "gross": gross,
        "cash": max(0.0, 1.0 - gross),
        "n_positions": len(weights),
        "regime_on": regime_on,
        "regime_state": diagnostics_regime,
        "regime_gross_multiplier": gross_mult,
        "kelly_scale": kelly_scale,
        "equity": float(equity),
        "hold_bucket": cfg.get("hold_bucket"),
    }
    return weights, diagnostics
