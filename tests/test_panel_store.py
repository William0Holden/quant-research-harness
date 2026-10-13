"""The backtest's fast vol path must equal the store's slow one.

`estimate_vols` opts into `store.annualized_vols` when the store offers it (PanelStore
does; PITStore doesn't). That is a performance shortcut on the SIZING path -- if the two
estimators disagree, the backtest sizes a different book than live trading does, and
every parameter chosen from it is chosen from the wrong portfolio.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from qts.backtest.panel_store import PanelStore
from qts.risk.sizing import estimate_vols


class _SlowOnly:
    """PanelStore with the fast path hidden -- forces estimate_vols down the per-symbol
    prices_as_of loop that PITStore uses in production."""

    def __init__(self, panel):
        self._inner = PanelStore(panel)

    def prices_as_of(self, symbols, as_of, lookback_days=None):
        return self._inner.prices_as_of(symbols, as_of, lookback_days)


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-01-01", periods=300)
    data = {}
    for i, sym in enumerate(["AAA", "BBB", "CCC"]):
        rets = rng.normal(0.0004, 0.01 * (i + 1), len(idx))
        data[sym] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def test_fast_and_slow_vol_estimators_agree(panel):
    as_of = date(2026, 1, 15)
    syms = ["AAA", "BBB", "CCC"]
    fast = estimate_vols(syms, PanelStore(panel), as_of, lookback=126)
    slow = estimate_vols(syms, _SlowOnly(panel), as_of, lookback=126)
    pd.testing.assert_series_equal(fast, slow, rtol=1e-9, check_names=False)


def test_vols_rank_by_true_volatility(panel):
    v = estimate_vols(["AAA", "BBB", "CCC"], PanelStore(panel), date(2026, 1, 15))
    assert v["AAA"] < v["BBB"] < v["CCC"]        # generated with rising vol


def test_insufficient_history_is_nan_not_zero(panel):
    """A name with too few bars must be DROPPED by the sizer, not handed a 0 vol --
    0 vol would make 1/vol infinite and hand it the entire book."""
    v = estimate_vols(["AAA"], PanelStore(panel), date(2025, 1, 3), lookback=126)
    assert np.isnan(v["AAA"])


def test_unknown_symbol_is_nan(panel):
    v = estimate_vols(["ZZZ"], PanelStore(panel), date(2026, 1, 15))
    assert np.isnan(v["ZZZ"])


def test_prices_as_of_never_sees_the_future(panel):
    px = PanelStore(panel).prices_as_of(["AAA"], date(2025, 6, 30))
    assert px["dt"].max() <= pd.Timestamp("2025-06-30")
