from datetime import date

import numpy as np
import pandas as pd
import pytest

from qts.backtest.engine import run_backtest
from qts.features.base import FeatureProvider
from qts.pit.store import PITStore
from qts.types import Backtestability, Direction, FeatureSpec


class UpProvider(FeatureProvider):
    """Bullish on high-index symbols (which drift up faster in the fixture)."""
    spec = FeatureSpec(name="momentum_12_1", direction=Direction.HIGHER_BETTER,
                       backtestability=Backtestability.BACKTESTABLE)

    def compute(self, store, symbols, as_of):
        return pd.Series({s: float(s[3:]) for s in symbols})


class DownProvider(FeatureProvider):
    """Exactly wrong."""
    spec = FeatureSpec(name="contrarian", direction=Direction.HIGHER_BETTER,
                       backtestability=Backtestability.BACKTESTABLE)

    def compute(self, store, symbols, as_of):
        return pd.Series({s: -float(s[3:]) for s in symbols})


@pytest.fixture
def seeded(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    syms = [f"SYM{i}" for i in range(10)]
    store.write_universe(pd.DataFrame({
        "symbol": syms, "cik": None, "name": syms,
        "active_from": date(2020, 1, 1), "active_to": None, "delisted": False,
    }))
    dts = pd.bdate_range("2024-01-01", "2025-06-30")
    rows = []
    for i, s in enumerate(syms):
        px = 100 * np.cumprod(np.full(len(dts), 1 + 0.001 * i))
        rows.extend({
            "symbol": s, "dt": d.date(), "close": p, "adj_close": p, "volume": 1000,
            "knowledge_time": d + pd.Timedelta(hours=23),
        } for d, p in zip(dts, px))
    store.write_prices(pd.DataFrame(rows))
    yield store
    store.close()


def test_weights_fn_beats_bad_static_weights(seeded):
    providers = [UpProvider(), DownProvider()]
    kw = dict(start=date(2025, 1, 1), end=date(2025, 6, 1))
    calls = []

    def all_up(as_of):
        calls.append(as_of)
        return {"momentum_12_1": 1.0, "contrarian": 0.0}

    good = run_backtest(seeded, providers, weights_fn=all_up, **kw)
    bad = run_backtest(seeded, providers,
                       weights={"momentum_12_1": 0.0, "contrarian": 1.0}, **kw)
    assert len(calls) > 5                       # called per rebalance date
    assert (good.periods["long_only_net"].mean()
            > bad.periods["long_only_net"].mean())


def test_weights_fn_none_falls_back(seeded):
    providers = [UpProvider()]
    kw = dict(start=date(2025, 1, 1), end=date(2025, 6, 1))
    a = run_backtest(seeded, providers, weights_fn=lambda d: None, **kw)
    b = run_backtest(seeded, providers, **kw)
    pd.testing.assert_frame_equal(a.periods, b.periods)
