"""Sizing + the sentiment overlay. The overlay may only ever REDUCE risk, and the GPU-down
path must degrade to the trend book rather than to an error."""
import numpy as np
import pandas as pd
import pytest

from qts.crypto.sizing import (build_crypto_targets, extremity, overlay_multiplier,
                               realized_vol, vol_scalar)


def _px(vol_daily, n=200, drift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol_daily, n))), index=idx)


def test_realized_vol_is_nan_not_zero_when_history_is_thin():
    """A zero vol makes the 1/vol scalar INFINITE and would hand one asset unbounded
    leverage. NaN forces the caller to refuse to size. Never 0.0."""
    assert np.isnan(realized_vol(_px(0.03, n=5)))
    assert np.isnan(realized_vol(pd.Series([100.0] * 60)))   # flat prices: sd == 0


def test_vol_scalar_never_levers_past_max_gross():
    assert vol_scalar(0.01, vol_target=0.40, max_gross=1.0) == 1.0   # calm -> capped, not 40x
    assert vol_scalar(0.80, vol_target=0.40) == pytest.approx(0.5)
    assert vol_scalar(float("nan")) == 0.0     # unknown risk -> hold nothing
    assert vol_scalar(0.0) == 0.0


def test_overlay_can_only_reduce_never_amplify():
    """The overlay is a RISK brake. If it could exceed 1.0 it would be a directional bet,
    which is precisely what the sentiment literature says does not work."""
    for z in (-10, -3, -1, 0, 1, 3, 10):
        assert 0.5 <= overlay_multiplier(float(z)) <= 1.0
    assert overlay_multiplier(0.0) == 1.0                    # neutral sentiment -> no effect
    assert overlay_multiplier(5.0) < overlay_multiplier(0.5)  # more extreme -> smaller book


def test_extremity_saturates():
    """Beyond 2 sigma an extreme reading is far more likely a bot swarm or a thin-data
    artefact than information -- 14-56% of crypto social accounts are bots."""
    assert extremity(2.0) == 1.0 and extremity(50.0) == 1.0
    assert extremity(float("nan")) == 0.0


def test_gpu_down_degrades_to_the_trend_book_not_to_an_error():
    """THE user-requested fallback. the GPU worker is shared; when it is unavailable the sentiment
    score is NaN and the book must trade on trend alone."""
    assert overlay_multiplier(None) == 1.0
    assert overlay_multiplier(float("nan")) == 1.0

    closes = {"BTC/USD": _px(0.03), "ETH/USD": _px(0.04, seed=2)}
    te = {"BTC/USD": 1.0, "ETH/USD": 1.0}
    w_down, d_down = build_crypto_targets(te, closes, sent_z=None, overlay_on=True)
    w_off, _ = build_crypto_targets(te, closes, sent_z=None, overlay_on=False)
    assert d_down["overlay_m"] == 1.0
    assert w_down == w_off, "GPU-down book must equal the trend-only book exactly"


def test_gross_is_hard_capped_and_never_levered():
    closes = {"BTC/USD": _px(0.001), "ETH/USD": _px(0.001, seed=2)}   # near-zero vol
    w, d = build_crypto_targets({"BTC/USD": 1.0, "ETH/USD": 1.0}, closes, vol_target=5.0)
    assert d["gross"] <= 1.0 + 1e-9, "a long-only coin book must never be levered"


def test_nan_trend_exposure_holds_nothing_rather_than_guessing():
    closes = {"BTC/USD": _px(0.03), "ETH/USD": _px(0.03, seed=2)}
    w, _ = build_crypto_targets({"BTC/USD": float("nan"), "ETH/USD": 1.0}, closes)
    assert "BTC/USD" not in w and "ETH/USD" in w


def test_overlay_attribution_is_exact():
    """r_book = m * r_trend. The record keeps BOTH books, so the overlay's contribution is
    arithmetic, not a story told afterwards."""
    closes = {"BTC/USD": _px(0.03), "ETH/USD": _px(0.04, seed=2)}
    te = {"BTC/USD": 1.0, "ETH/USD": 1.0}
    w_on, d_on = build_crypto_targets(te, closes, sent_z=2.0, overlay_on=True)
    w_off, d_off = build_crypto_targets(te, closes, sent_z=2.0, overlay_on=False)
    m = d_on["overlay_m"]
    assert m < 1.0
    for s in w_off:
        assert w_on[s] == pytest.approx(w_off[s] * m)
    assert d_on["trend_gross"] == pytest.approx(d_off["gross"])
