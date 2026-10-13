"""Sizing / risk layer for the long-only live book.

Converts ranker output into hard-limited, vol-targeted, regime-conditioned target
weights (convertible to exec.broker.TargetPortfolio).
"""
from __future__ import annotations

from .limits import KillSwitch, apply_limits
from .portfolio import build_target_portfolio
from .regime import DEGRADED_GROSS_MULTIPLIER, fetch_fred_series, regime_state
from .sizing import (
    build_long_only,
    estimate_vols,
    portfolio_vol,
    quarter_kelly_scale,
    vol_target_weights,
)

__all__ = [
    "build_target_portfolio",
    "build_long_only",
    "vol_target_weights",
    "quarter_kelly_scale",
    "estimate_vols",
    "portfolio_vol",
    "apply_limits",
    "KillSwitch",
    "regime_state",
    "fetch_fred_series",
    "DEGRADED_GROSS_MULTIPLIER",
]
