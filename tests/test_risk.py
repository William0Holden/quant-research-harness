"""Pure-logic tests for the sizing/risk layer. NO network required.

The single live FRED smoke test lives in src/qts/risk/regime.py's __main__ block.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qts.risk.limits import KillSwitch, apply_limits
from qts.risk.sizing import (
    build_long_only,
    portfolio_vol,
    quarter_kelly_scale,
    vol_target_weights,
)


# --------------------------------------------------------------------------- #
# Fake PIT store: serves synthetic price history without DuckDB / network.
# --------------------------------------------------------------------------- #
class FakeStore:
    """Mimics PITStore.prices_as_of for a set of symbols with given daily vols."""

    def __init__(self, vols: dict[str, float], n: int = 300, seed: int = 0):
        self.frames: dict[str, pd.DataFrame] = {}
        rng = np.random.default_rng(seed)
        end = date(2024, 1, 1)
        dts = pd.bdate_range(end=end, periods=n)
        for sym, dv in vols.items():
            rets = rng.normal(0.0, dv, size=n)
            price = 100.0 * np.exp(np.cumsum(rets))
            self.frames[sym] = pd.DataFrame(
                {"symbol": sym, "dt": dts, "close": price, "adj_close": price,
                 "volume": 1_000_000}
            )

    def prices_as_of(self, symbols, as_of, lookback_days=None):
        out = []
        for s in symbols:
            if s in self.frames:
                out.append(self.frames[s])
        if not out:
            return pd.DataFrame(columns=["symbol", "dt", "close", "adj_close", "volume"])
        return pd.concat(out, ignore_index=True)


AS_OF = date(2024, 1, 1)


# --------------------------------------------------------------------------- #
# vol_target_weights
# --------------------------------------------------------------------------- #
def test_vol_target_weights_inverse_vol_ordering():
    # lower-vol name should get a HIGHER weight than higher-vol name
    store = FakeStore({"LOWV": 0.005, "HIGHV": 0.03})
    w = vol_target_weights(["LOWV", "HIGHV"], store, AS_OF, target_vol=0.10)
    assert w["LOWV"] > w["HIGHV"]


def test_vol_target_weights_normalization_and_scaling():
    store = FakeStore({"A": 0.01, "B": 0.012, "C": 0.011})
    w = vol_target_weights(["A", "B", "C"], store, AS_OF, target_vol=0.10)
    # gross must be <= 1 (unlevered long book)
    assert 0 < w.sum() <= 1.0 + 1e-9
    # the pre-scale shape is normalized inverse-vol -> after scaling, relative
    # proportions preserved and all positive
    assert (w > 0).all()


def test_vol_target_weights_hits_target_when_scaled():
    # very high-vol single name -> scale should bite and portfolio vol ~= target
    store = FakeStore({"X": 0.04})  # ~63% annualized -> way above 10% target
    w = vol_target_weights(["X"], store, AS_OF, target_vol=0.10)
    from qts.risk.sizing import estimate_vols
    vols = estimate_vols(["X"], store, AS_OF)
    pv = portfolio_vol(w, vols)
    assert pv == pytest.approx(0.10, rel=0.05)
    assert w.sum() < 1.0  # had to scale down


def test_vol_target_weights_drops_insufficient_history():
    store = FakeStore({"GOOD": 0.01}, n=300)
    store.frames["SHORT"] = store.frames["GOOD"].head(5).assign(symbol="SHORT")
    w = vol_target_weights(["GOOD", "SHORT"], store, AS_OF)
    assert "GOOD" in w.index
    assert "SHORT" not in w.index


# --------------------------------------------------------------------------- #
# quarter_kelly_scale  (the half-edge contract)
# --------------------------------------------------------------------------- #
def test_quarter_kelly_uses_half_edge():
    # f = 0.25 * ((edge/2) / var)
    edge, var = 0.08, 0.04
    expected = 0.25 * ((edge / 2.0) / var)
    assert quarter_kelly_scale(edge, var) == pytest.approx(expected)


def test_quarter_kelly_is_half_of_full_quarter_kelly():
    # sizing off half the edge must give exactly half of what full-edge quarter-kelly would
    edge, var = 0.10, 0.05
    half_edge = quarter_kelly_scale(edge, var)
    full_edge = 0.25 * (edge / var)
    assert half_edge == pytest.approx(full_edge / 2.0)


def test_quarter_kelly_clips_to_unit_interval():
    assert quarter_kelly_scale(10.0, 0.001) == 1.0  # huge -> clipped to 1
    assert quarter_kelly_scale(-0.05, 0.04) == 0.0  # negative edge -> 0
    assert quarter_kelly_scale(0.05, 0.0) == 0.0    # zero variance -> 0
    assert quarter_kelly_scale(0.05, -1.0) == 0.0   # negative variance -> 0


# --------------------------------------------------------------------------- #
# build_long_only
# --------------------------------------------------------------------------- #
def _ranked(buckets: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"score": {k: 0.0 for k in buckets}, "bucket": buckets})


def test_build_long_only_selects_top_bucket():
    store = FakeStore({"TOP1": 0.01, "TOP2": 0.012, "MID": 0.01, "BOT": 0.01})
    ranked = _ranked({"TOP1": 5.0, "TOP2": 5.0, "MID": 3.0, "BOT": 1.0})
    out = build_long_only(ranked, store, AS_OF)
    assert set(out) == {"TOP1", "TOP2"}
    assert all(v > 0 for v in out.values())


def test_build_long_only_applies_kelly_gross():
    store = FakeStore({"A": 0.01, "B": 0.011})
    ranked = _ranked({"A": 5.0, "B": 5.0})
    full = build_long_only(ranked, store, AS_OF)
    scaled = build_long_only(
        ranked, store, AS_OF, cfg={"expected_edge": 0.05, "variance": 0.10}
    )
    ks = quarter_kelly_scale(0.05, 0.10)
    assert sum(scaled.values()) == pytest.approx(sum(full.values()) * ks, rel=1e-6)


def test_build_long_only_empty_inputs():
    store = FakeStore({"A": 0.01})
    assert build_long_only(pd.DataFrame(), store, AS_OF) == {}
    assert build_long_only(_ranked({"A": float("nan")}), store, AS_OF) == {}


# --------------------------------------------------------------------------- #
# apply_limits
# --------------------------------------------------------------------------- #
def test_apply_limits_caps_position():
    w = {"BIG": 0.5, "A": 0.25, "B": 0.25}
    out = apply_limits(w, max_position=0.10)
    assert all(v <= 0.10 + 1e-9 for v in out.values())


def test_apply_limits_preserves_gross_when_feasible():
    w = {"A": 0.4, "B": 0.3, "C": 0.3}  # gross 1.0, n=3, cap .5 feasible
    out = apply_limits(w, max_position=0.5)
    assert sum(out.values()) == pytest.approx(1.0, rel=1e-9)
    assert max(out.values()) <= 0.5 + 1e-9


def test_apply_limits_caps_out_to_cash_when_infeasible():
    # cap 0.1 * 3 names = 0.3 max achievable gross
    w = {"A": 0.4, "B": 0.3, "C": 0.3}
    out = apply_limits(w, max_position=0.10)
    assert sum(out.values()) == pytest.approx(0.30, rel=1e-9)
    assert all(v == pytest.approx(0.10) for v in out.values())


def test_apply_limits_max_names():
    w = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    out = apply_limits(w, max_position=1.0, max_names=2)
    assert set(out) == {"A", "B"}
    assert sum(out.values()) == pytest.approx(0.7, rel=1e-9)  # gross preserved


def test_apply_limits_drops_nonpositive():
    out = apply_limits({"A": 0.5, "B": -0.1, "C": 0.0}, max_position=1.0)
    assert set(out) == {"A"}


# --------------------------------------------------------------------------- #
# KillSwitch
# --------------------------------------------------------------------------- #
def test_killswitch_trips_on_drawdown():
    ks = KillSwitch(max_drawdown=0.25)
    curve = [100, 110, 120, 100, 85]  # peak 120 -> 85 = ~29% dd
    assert ks.check(curve) is True
    assert ks.tripped is True


def test_killswitch_no_trip_under_threshold():
    ks = KillSwitch(max_drawdown=0.25)
    curve = [100, 110, 120, 100]  # 120 -> 100 = ~16.7% dd
    assert ks.check(curve) is False
    assert ks.tripped is False


def test_killswitch_latches():
    ks = KillSwitch(max_drawdown=0.25)
    ks.check([100, 120, 80])  # trips
    assert ks.tripped is True
    # full recovery should NOT un-trip
    assert ks.check([100, 120, 80, 200]) is True


def test_killswitch_custom_threshold_and_reset():
    ks = KillSwitch(max_drawdown=0.10)
    assert ks.check([100, 95, 89]) is True  # 11% dd
    ks.reset()
    assert ks.tripped is False
    assert ks.check([100, 95]) is False


def test_killswitch_rejects_bad_threshold():
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=0.0)
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=1.5)


# --------------------------------------------------------------------------- #
# portfolio.build_target_portfolio (regime_off path -> no network)
# --------------------------------------------------------------------------- #
def test_build_target_portfolio_no_regime():
    from qts.risk.portfolio import build_target_portfolio

    store = FakeStore({f"S{i}": 0.01 + 0.001 * i for i in range(6)})
    ranked = _ranked({f"S{i}": 5.0 for i in range(6)})
    weights, diag = build_target_portfolio(
        ranked, store, AS_OF, equity=100_000, regime_on=False,
        cfg={"max_position": 0.30},
    )
    assert all(v <= 0.30 + 1e-9 for v in weights.values())
    assert diag["gross"] <= 1.0 + 1e-9
    assert diag["n_positions"] == len(weights)
    assert diag["regime_state"] is None  # regime off


# regime_state resilience: a failed FRED fetch must degrade, never raise
# (2026-07-03 incident: transient hy_oas failure -> empty-Series RangeIndex vs
# Timestamp TypeError -> outer catch -> brake silently failed open)
#
# FAIL-CLOSED as of 2026-07-28. These tests used to assert gross_multiplier == 1.0
# on an unreadable series ("fail-open is the documented policy"). The 2026-07-24
# rebalance ran at full gross with all three series NaN, which is the failure mode
# that policy invites. Unknown regime now caps gross at DEGRADED_GROSS_MULTIPLIER.

@pytest.fixture(autouse=True)
def _isolate_regime_cache(tmp_path, monkeypatch):
    """Point the last-known-good cache at a temp file.

    Without this, a developer box with a populated outputs/regime_cache.json would
    serve real data into the "FRED is down" tests and they would pass for the wrong
    reason -- the exact class of false green this fix exists to prevent.
    """
    monkeypatch.setenv("QTS_REGIME_CACHE", str(tmp_path / "regime_cache.json"))


def test_regime_state_survives_failed_hy_fetch(monkeypatch):
    from qts.risk import regime as R

    def flaky(sid, as_of, timeout=20.0):
        if sid == "BAMLH0A0HYM2":
            raise ConnectionError("FRED down")
        idx = pd.date_range("2026-06-24", periods=10, freq="D")
        return pd.Series(15.0, index=idx, name=sid)

    monkeypatch.setattr(R, "fetch_fred_series", flaky)
    state = R.regime_state(date(2026, 7, 3))
    # hy_oas is a DECISION series: losing it means we cannot evaluate credit stress.
    assert state["gross_multiplier"] == R.DEGRADED_GROSS_MULTIPLIER
    assert state["degraded"] is True
    assert pd.isna(state["hy_oas"])
    assert state["fetch_errors"] == ["hy_oas"]     # degraded state is VISIBLE
    assert state["series_source"]["hy_oas"] == "missing"
    assert state["series_source"]["vix"] == "live"


def test_regime_state_survives_all_fetches_failing(monkeypatch):
    from qts.risk import regime as R

    def dead(sid, as_of, timeout=20.0):
        raise ConnectionError("FRED down")

    monkeypatch.setattr(R, "fetch_fred_series", dead)
    state = R.regime_state(date(2026, 7, 3))
    assert state["gross_multiplier"] == R.DEGRADED_GROSS_MULTIPLIER   # fail CLOSED
    assert state["degraded"] is True
    assert sorted(state["fetch_errors"]) == ["hy_oas", "term_spread", "vix"]


def test_term_spread_failure_alone_does_not_degrade(monkeypatch):
    """term_spread feeds no tripwire, so losing it must NOT de-risk the book."""
    from qts.risk import regime as R

    def flaky(sid, as_of, timeout=20.0):
        if sid == "T10Y2Y":
            raise ConnectionError("FRED down")
        idx = pd.date_range("2026-05-01", periods=64, freq="D")
        # HY well below its own q75, VIX well below the spike floor -> risk-on.
        val = 3.0 if sid == "BAMLH0A0HYM2" else 15.0
        s = pd.Series(val, index=idx, name=sid)
        if sid == "BAMLH0A0HYM2":
            s.iloc[:-1] = 4.0          # last obs below the trailing quantile
        return s

    monkeypatch.setattr(R, "fetch_fred_series", flaky)
    state = R.regime_state(date(2026, 7, 3))
    assert state["fetch_errors"] == ["term_spread"]
    assert state["degraded"] is False
    assert state["gross_multiplier"] == 1.0


def test_cache_covers_a_transient_outage(monkeypatch):
    """A hiccup must change NOTHING: the cached history keeps the brake fully evaluable."""
    from qts.risk import regime as R

    idx = pd.date_range("2026-05-01", periods=64, freq="D")

    def healthy(sid, as_of, timeout=20.0):
        val = 3.0 if sid == "BAMLH0A0HYM2" else 15.0
        s = pd.Series(val, index=idx, name=sid)
        if sid == "BAMLH0A0HYM2":
            s.iloc[:-1] = 4.0
        return s

    monkeypatch.setattr(R, "fetch_fred_series", healthy)
    good = R.regime_state(date(2026, 7, 3))
    assert good["degraded"] is False

    def dead(sid, as_of, timeout=20.0):
        raise ConnectionError("FRED down")

    monkeypatch.setattr(R, "fetch_fred_series", dead)
    fallback = R.regime_state(date(2026, 7, 3))
    assert fallback["series_source"]["hy_oas"] == "cache"
    assert fallback["degraded"] is False                      # cache kept it evaluable
    assert fallback["gross_multiplier"] == good["gross_multiplier"]
    assert fallback["fetch_errors"] == ["hy_oas", "term_spread", "vix"]  # still visible


def test_stale_cache_degrades(monkeypatch):
    """A prolonged blackout is different from a hiccup: past the window, we don't know."""
    from qts.risk import regime as R

    idx = pd.date_range("2026-05-01", periods=64, freq="D")

    def healthy(sid, as_of, timeout=20.0):
        return pd.Series(15.0, index=idx, name=sid)

    monkeypatch.setattr(R, "fetch_fred_series", healthy)
    R.regime_state(date(2026, 7, 3))       # populate the cache (last obs 2026-07-03)

    def dead(sid, as_of, timeout=20.0):
        raise ConnectionError("FRED down")

    monkeypatch.setattr(R, "fetch_fred_series", dead)
    fresh = R.regime_state(date(2026, 7, 10))    # 7d old cache -> inside the window
    assert fresh["degraded"] is False
    stale = R.regime_state(date(2026, 7, 20))    # 17d old cache -> beyond it
    assert stale["degraded"] is True
    assert "hy_oas" in stale["stale_series"] and "vix" in stale["stale_series"]
    assert stale["gross_multiplier"] == R.DEGRADED_GROSS_MULTIPLIER


def test_build_target_portfolio_fails_closed_when_regime_raises(monkeypatch):
    """The last fail-open path: regime_state itself blowing up must not size at 1.0."""
    import qts.risk.portfolio as P

    def boom(as_of, series=None):
        raise RuntimeError("regime layer unavailable")

    monkeypatch.setattr(P, "regime_state", boom)
    store = FakeStore({f"S{i}": 0.01 + 0.001 * i for i in range(6)})
    ranked = _ranked({f"S{i}": 5.0 for i in range(6)})
    weights, diag = P.build_target_portfolio(
        ranked, store, AS_OF, equity=100_000, regime_on=True,
        cfg={"max_position": 0.30},
    )
    assert diag["regime_gross_multiplier"] == P.DEGRADED_GROSS_MULTIPLIER
    assert diag["regime_state"]["degraded"] is True
    assert sum(weights.values()) <= 1.0


def test_fetch_retry_recovers_transient_failure(monkeypatch):
    from qts.risk import regime as R
    calls = {"n": 0}

    def flaky_once(sid, as_of, timeout=20.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        idx = pd.date_range("2026-06-01", periods=40, freq="D")
        return pd.Series(3.0, index=idx, name=sid)

    monkeypatch.setattr(R, "fetch_fred_series", flaky_once)
    s = R._fetch_with_retry("BAMLH0A0HYM2", date(2026, 7, 3), sleep_s=0)
    assert len(s) == 40 and calls["n"] == 2
