from datetime import date

import numpy as np
import pandas as pd
import pytest

from qts.backtest.engine import _trading_days, make_schedule
from qts.features.base import FeatureProvider
from qts.pit.store import PITStore
from qts.research.ic import ic_summary, make_weights_fn, provider_ic, trailing_ic_weights
from qts.types import Backtestability, Direction, FeatureSpec


class RankOracle(FeatureProvider):
    """Feature = symbol index i; returns drift with i -> IC should be ~+1."""
    spec = FeatureSpec(name="oracle", direction=Direction.HIGHER_BETTER,
                       backtestability=Backtestability.BACKTESTABLE)

    def compute(self, store, symbols, as_of):
        return pd.Series({s: float(s[3:]) for s in symbols})


class NoiseProvider(FeatureProvider):
    spec = FeatureSpec(name="noise", direction=Direction.HIGHER_BETTER,
                       backtestability=Backtestability.BACKTESTABLE)

    def compute(self, store, symbols, as_of):
        rng = np.random.default_rng(hash(as_of) % 2**32)
        return pd.Series(rng.standard_normal(len(symbols)), index=symbols)


@pytest.fixture
def seeded(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    syms = [f"SYM{i}" for i in range(10)]
    store.write_universe(pd.DataFrame({
        "symbol": syms, "cik": None, "name": syms,
        "active_from": date(2020, 1, 1), "active_to": None, "delisted": False,
    }))
    dts = pd.bdate_range("2024-06-01", "2025-06-30")
    rows = []
    for i, s in enumerate(syms):
        px = 100 * np.cumprod(np.full(len(dts), 1 + 0.001 * i))  # monotone drift in i
        rows.extend({
            "symbol": s, "dt": d.date(), "close": p, "adj_close": p, "volume": 1000,
            "knowledge_time": d + pd.Timedelta(hours=23),
        } for d, p in zip(dts, px))
    store.write_prices(pd.DataFrame(rows))
    yield store
    store.close()


def test_oracle_has_high_ic_noise_does_not(seeded):
    icdf = provider_ic(seeded, [RankOracle(), NoiseProvider()],
                       date(2025, 1, 1), date(2025, 6, 1))
    assert icdf["oracle"].mean() > 0.9
    assert abs(icdf["noise"].mean()) < 0.5
    summ = ic_summary(icdf)
    assert summ.loc["oracle", "t_stat"] > 3
    assert summ.loc["oracle", "n_periods"] >= 15


def test_ic_rows_indexed_by_exit_bar(seeded):
    icdf = provider_ic(seeded, [RankOracle()], date(2025, 1, 1), date(2025, 6, 1))
    schedule = make_schedule(_trading_days(seeded), date(2025, 1, 1), date(2025, 6, 1), "W-FRI")
    for k in range(len(icdf)):
        assert icdf.index[k] > pd.Timestamp(schedule[k])


def test_trailing_weights_are_past_only_and_normalized():
    idx = pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"])
    icdf = pd.DataFrame({"a": [0.5, 0.5, 99.0], "b": [-0.1, -0.1, 99.0]}, index=idx)
    w = trailing_ic_weights(icdf, date(2025, 1, 17), window=26)
    assert w == {"a": 1.0, "b": 0.0}      # 99s on 01-17 unseen; negative floored to 0
    assert trailing_ic_weights(icdf, date(2025, 1, 3)) is None   # no history yet


def test_make_weights_fn_wraps_trailing(seeded):
    icdf = pd.DataFrame({"a": [0.4, 0.6]},
                        index=pd.to_datetime(["2025-01-03", "2025-01-10"]))
    fn = make_weights_fn(icdf)
    assert fn(date(2025, 1, 17)) == {"a": 1.0}
    assert fn(date(2025, 1, 3)) is None


def test_trailing_weights_all_nan_column_gets_zero():
    idx = pd.to_datetime(["2025-01-03", "2025-01-10"])
    icdf = pd.DataFrame({"a": [0.5, 0.9], "b": [np.nan, np.nan]}, index=idx)
    w = trailing_ic_weights(icdf, date(2025, 1, 17), window=26)
    assert w == {"a": 1.0, "b": 0.0}
    assert not any(np.isnan(v) for v in w.values())
