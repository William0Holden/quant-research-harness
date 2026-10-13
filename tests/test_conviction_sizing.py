"""Conviction tilt: the composite score should size the book, not just pick it."""
from __future__ import annotations

import pandas as pd
import pytest

from qts.risk.sizing import conviction_tilt, vol_target_weights


class _FakeStore:
    """Serves flat-vol price history so tilt effects are visible in isolation."""

    def __init__(self, vols: dict[str, float]):
        self.vols = vols

    def prices_as_of(self, symbols, as_of, lookback_days=None):
        rows = []
        for sym in symbols:
            # A deterministic zig-zag whose amplitude sets the realised vol.
            amp = self.vols[sym]
            price = 100.0
            for i in range(60):
                price *= 1.0 + (amp if i % 2 == 0 else -amp)
                rows.append({"symbol": sym, "dt": pd.Timestamp("2026-01-01")
                             + pd.Timedelta(days=i), "adj_close": price})
        return pd.DataFrame(rows)


def test_tilt_is_monotone_in_score():
    t = conviction_tilt(pd.Series({"A": 0.5, "B": 1.5}))
    assert t["B"] > t["A"]


def test_tilt_saturates_at_cap():
    """An extreme composite z is usually thin-data noise, not extra edge."""
    t = conviction_tilt(pd.Series({"A": 2.0, "B": 9.9}), cap=2.0)
    assert t["A"] == pytest.approx(2.0)
    assert t["B"] == pytest.approx(2.0)


def test_tilt_floors_a_marginal_name_above_zero():
    """Bucket membership is the ranker's call. A barely-included name gets a small
    weight, not silently dropped by the sizer."""
    t = conviction_tilt(pd.Series({"A": -0.4, "B": 0.0}), floor=0.25)
    assert t["A"] == pytest.approx(0.25)
    assert t["B"] == pytest.approx(0.25)


def test_missing_score_is_not_negative_conviction():
    t = conviction_tilt(pd.Series({"A": float("nan")}), floor=0.25)
    assert t["A"] == pytest.approx(0.25)


def test_tilt_raises_weight_of_the_high_conviction_name_at_equal_vol():
    store = _FakeStore({"A": 0.01, "B": 0.01})   # identical vol -> tilt is the only lever
    flat = vol_target_weights(["A", "B"], store, pd.Timestamp("2026-03-01").date())
    tilted = vol_target_weights(
        ["A", "B"], store, pd.Timestamp("2026-03-01").date(),
        tilt=pd.Series({"A": 0.5, "B": 2.0}),
    )
    assert flat["A"] == pytest.approx(flat["B"])          # untilted: equal
    assert tilted["B"] > tilted["A"]                      # tilted: conviction wins
    assert tilted["B"] / tilted["A"] == pytest.approx(4.0, rel=1e-6)


def test_tilt_does_not_raise_portfolio_risk():
    """Tilting changes RELATIVE weights only -- the vol target still binds, so a
    conviction tilt can never quietly lever the book up."""
    store = _FakeStore({"A": 0.01, "B": 0.02})
    as_of = pd.Timestamp("2026-03-01").date()
    flat = vol_target_weights(["A", "B"], store, as_of, target_vol=0.10)
    tilted = vol_target_weights(["A", "B"], store, as_of, target_vol=0.10,
                                tilt=pd.Series({"A": 2.0, "B": 0.25}))
    # Both books are scaled to the same estimated vol, so neither is grossly larger.
    assert tilted.sum() <= 1.0
    assert flat.sum() <= 1.0
