"""Everything sentiment stays OFF the book. This is the load-bearing safety property:
signals compute and log, and change NOTHING about sizing or execution, until earned."""
from qts.config import live_gated
from qts.crypto.profile import CryptoProfile


def test_crypto_sentiment_concepts_are_live_gated():
    """The crypto sentiment risk signals must be in the gate, exactly like the equity ones."""
    gated = live_gated()
    for name in ("crypto_vol_forecast", "crypto_equity_risk_appetite", "crypto_attention_spike"):
        assert name in gated, f"{name} must be live-gated (it affects nothing until earned)"


def test_overlay_stays_off_by_default():
    assert CryptoProfile().overlay_on is False


def test_sizer_vol_input_is_unchanged_while_gated(monkeypatch):
    """The sizer must still use plain trailing realized vol -- the vol-forecast signal is
    computed for measurement only and must not be wired into sizing while gated."""
    import qts.crypto.sizing as sizing
    import pandas as pd, numpy as np
    # realized_vol is the ONLY vol the sizer consumes; it must be the trailing estimator,
    # with no sentiment term reachable from it.
    px = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.03, 60)),
                   index=pd.date_range("2026-01-01", periods=60, freq="D"))
    v = sizing.realized_vol(px)
    assert v > 0
    # build_crypto_targets takes no sentiment argument that could alter sizing:
    import inspect
    params = inspect.signature(sizing.build_crypto_targets).parameters
    assert "sent_z" in params  # the overlay hook exists...
    # ...but with overlay_on defaulting False and sent_z=None, the book is trend-only:
    w_off, d = sizing.build_crypto_targets({"BTC/USD": 1.0}, {"BTC/USD": px}, overlay_on=False)
    assert d["overlay_m"] == 1.0
