"""The three risk signals. Each must: be NaN on thin data, be monotone in its input, be
bounded, and never fabricate a zero. All grade against RISK, never returns."""
import numpy as np
import pandas as pd
import pytest

from qts.crypto.sentiment import signals as sg


def _counts(vals):
    return pd.Series(vals, index=pd.date_range("2026-01-01", periods=len(vals), freq="D"))


def test_attention_z_needs_a_baseline():
    assert np.isnan(sg.attention_z(_counts([5, 6, 7])))            # far too short
    z = sg.attention_z(_counts([5] * 30 + [50, 50, 50]))          # steady then a surge
    assert np.isfinite(z) and z > 2                               # clearly elevated


def test_attention_z_is_zero_ish_on_steady_volume():
    # Steady volume -> ~zero anomaly. The Poisson std floor makes a flat count series a
    # DEFINED zero (not NaN): a calm tape is "no surge", not "no opinion".
    z = sg.attention_z(_counts([10] * 40))
    assert np.isfinite(z) and abs(z) < 0.5
    noisy = _counts(list(10 + np.sin(np.arange(40))))
    assert np.isfinite(sg.attention_z(noisy))


def test_vol_forecast_raises_with_attention_and_is_floored():
    base = 0.50
    assert sg.vol_forecast(base, 0.0) == pytest.approx(base)      # neutral attention -> unchanged
    assert sg.vol_forecast(base, 2.0) > base                      # attention -> higher vol forecast
    assert sg.vol_forecast(base, -2.0) < base                     # quiet -> slightly lower
    assert sg.vol_forecast(base, -100.0) >= base * 0.25           # floored, never collapses
    # monotone in attention
    zs = [-2, -1, 0, 1, 2, 3]
    fs = [sg.vol_forecast(base, z) for z in zs]
    assert fs == sorted(fs)


def test_vol_forecast_no_opinion_paths():
    assert sg.vol_forecast(0.5, float("nan")) == pytest.approx(0.5)   # GPU down -> trailing vol
    assert np.isnan(sg.vol_forecast(float("nan"), 1.0))               # no vol -> no forecast
    assert np.isnan(sg.vol_forecast(0.0, 1.0))


def test_equity_risk_appetite_ignores_nans():
    s = pd.Series({"AAPL": 0.2, "MSFT": -0.1, "NVDA": float("nan")})
    assert sg.equity_risk_appetite(s) == pytest.approx(0.05)      # mean of the two present
    assert np.isnan(sg.equity_risk_appetite(pd.Series([np.nan, np.nan])))


def test_equity_risk_appetite_change_is_the_move():
    hist = pd.Series(range(10), index=pd.date_range("2026-01-01", periods=10, freq="D"),
                     dtype=float)
    assert sg.equity_risk_appetite_change(hist, window=5) == pytest.approx(5.0)
    assert np.isnan(sg.equity_risk_appetite_change(hist.iloc[:3], window=5))


def test_attention_spike_flags_only_real_surges():
    steady = _counts([8] * 30 + list(8 + np.sin(np.arange(5))))
    assert sg.attention_spike(steady)["spike"] is False
    surge = _counts([8] * 30 + [8, 9, 60])
    out = sg.attention_spike(surge)
    assert out["spike"] is True and out["attention_z"] >= 2.0
    # thin data is not an event
    assert sg.attention_spike(_counts([1, 2, 3]))["spike"] is False


def test_forward_risk_targets_have_no_lookahead():
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    prices = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.03, 40)),
                       index=idx)
    as_of = idx[25]
    fv = sg.forward_realized_vol(prices, as_of, horizon=10)
    assert np.isfinite(fv) and fv > 0
    # only future bars feed it: truncating the future makes it NaN, truncating the past doesn't
    assert np.isnan(sg.forward_realized_vol(prices.iloc[:28], as_of, horizon=10))
    dd = sg.forward_drawdown(prices, as_of, horizon=10)
    assert dd <= 0
    assert np.isnan(sg.forward_drawdown(prices, idx[-3], horizon=10))   # window runs off the end
