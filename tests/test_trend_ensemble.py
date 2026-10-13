"""The trend ensemble. Pins the two things that would silently break the strategy:
warmup must be NaN (not 0.0), and there must be no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from qts.crypto.trend import RULES, WARMUP, exposure, exposure_as_of, rule_signals


def _series(vals):
    idx = pd.date_range("2021-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=idx, dtype=float)


def test_exposure_is_bounded_and_monotone_in_rules_long():
    up = _series(np.linspace(100, 400, 400))     # relentless uptrend -> all rules long
    e = exposure(up)
    assert e.iloc[-1] == 1.0
    down = _series(np.linspace(400, 100, 400))   # relentless downtrend -> all rules flat
    assert exposure(down).iloc[-1] == 0.0
    assert ((e.dropna() >= 0) & (e.dropna() <= 1)).all()


def test_exposure_takes_fractional_values():
    """The whole point of an ensemble: it scales in and out rather than flipping. A book
    that could only ever be 0 or 1 would be a single rule wearing a costume."""
    rng = np.random.default_rng(0)
    px = _series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.04, 600))))
    vals = set(exposure(px).dropna().round(6))
    assert any(0.0 < v < 1.0 for v in vals), "ensemble never took an intermediate exposure"


def test_warmup_is_nan_never_zero():
    """NaN and 0.0 are DIFFERENT STATES. NaN = 'no opinion yet'; 0.0 = 'deliberately flat'.
    Conflating them backtests a cash position as a trend decision. A comparison against NaN
    silently yields False in pandas, so this is a real trap, not a hypothetical."""
    px = _series(np.linspace(100, 200, 400))
    e = exposure(px)
    assert e.iloc[: WARMUP - 1].isna().all()
    assert not np.isnan(e.iloc[-1])
    sig = rule_signals(px)
    assert list(sig.columns) == list(RULES)
    assert sig.iloc[0].isna().all()


def test_no_lookahead_truncating_the_future_cannot_change_the_past():
    """The decisive PIT test: the exposure computed at date d must be identical whether or
    not the series contains data after d."""
    rng = np.random.default_rng(7)
    px = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 800))))
    cut = 600
    full = exposure(px).iloc[cut]
    truncated = exposure(px.iloc[: cut + 1]).iloc[-1]
    assert full == truncated


def test_exposure_as_of_agrees_with_the_vectorised_path():
    """The live path calls exposure_as_of; the backtest calls exposure. If they disagree,
    the simulated book and the traded book are different strategies -- the exact bug class
    that made the equity backtest simulate a portfolio nobody owned."""
    rng = np.random.default_rng(3)
    px = _series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 500))))
    vec = exposure(px)
    for i in (300, 400, 499):
        assert exposure_as_of(px, px.index[i]) == pytest.approx(vec.iloc[i])


def test_exposure_as_of_is_nan_before_warmup():
    px = _series(np.linspace(100, 150, 300))
    assert np.isnan(exposure_as_of(px, px.index[50]))
