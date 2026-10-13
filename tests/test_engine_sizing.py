"""The sized engine path: it must simulate the LIVE book, and its shortcuts must be free.

`run_backtest(sizing=True)` is the default because the legacy equal-weight path reports
the returns of a portfolio nobody owns. These tests pin the two properties that make the
sized path trustworthy: it reflects the real sizing/turnover machinery, and the
performance shortcut used by the parameter sweep does not change any number it produces.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from qts.backtest.engine import run_backtest
from qts.features.base import FeatureProvider
from qts.features.ranker import rank_universe
from qts.pit.store import PITStore
from qts.types import Backtestability, Direction, FeatureSpec

KW = dict(start=date(2025, 1, 1), end=date(2025, 6, 1), regime_on=False)


class UpProvider(FeatureProvider):
    spec = FeatureSpec(name="momentum_12_1", direction=Direction.HIGHER_BETTER,
                       backtestability=Backtestability.BACKTESTABLE)

    def compute(self, store, symbols, as_of):
        return pd.Series({s: float(s[3:]) for s in symbols})


@pytest.fixture
def seeded(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    syms = [f"SYM{i}" for i in range(20)]
    store.write_universe(pd.DataFrame({
        "symbol": syms, "cik": None, "name": syms,
        "active_from": date(2020, 1, 1), "active_to": None, "delisted": False,
    }))
    dts = pd.bdate_range("2024-01-01", "2025-06-30")
    rng = np.random.default_rng(3)
    rows = []
    for i, s in enumerate(syms):
        # Drift rises with index (so UpProvider is right); vol also rises, so
        # inverse-vol weighting has something to actually do.
        rets = rng.normal(0.0002 * i, 0.004 * (1 + i / 10), len(dts))
        px = 100 * np.exp(np.cumsum(rets))
        rows.extend({
            "symbol": s, "dt": d.date(), "close": p, "adj_close": p, "volume": 1000,
            "knowledge_time": d + pd.Timedelta(hours=23),
        } for d, p in zip(dts, px))
    store.write_prices(pd.DataFrame(rows))
    yield store
    store.close()


def test_sized_book_is_not_the_equal_weight_book(seeded):
    """The whole reason this work exists: these are different portfolios. If they ever
    agree, the sized path has silently stopped applying the sizing pipeline."""
    sized = run_backtest(seeded, [UpProvider()], sizing=True, **KW)
    legacy = run_backtest(seeded, [UpProvider()], sizing=False, **KW)

    assert (sized.periods["gross_exposure"] < 1.0).any()      # vol target holds cash
    assert legacy.periods["gross_exposure"].max() == pytest.approx(1.0)
    assert not np.allclose(sized.periods["long_only_net"],
                           legacy.periods["long_only_net"])


def test_sized_weights_are_not_uniform(seeded):
    """Inverse-vol sizing must actually differentiate names."""
    res = run_backtest(seeded, [UpProvider()], sizing=True, **KW)
    last = list(res.weights_history.values())[-1]
    assert len(set(round(w, 6) for w in last.values())) > 1


def test_band_reduces_turnover(seeded):
    """The point of the band. Also the guard against it being wired up but inert."""
    no_band = run_backtest(seeded, [UpProvider()], sizing=True,
                           policy={"band_w": 0.0, "lam": 1.0, "dust_w": 0.0}, **KW)
    banded = run_backtest(seeded, [UpProvider()], sizing=True,
                          policy={"band_w": 0.002, "lam": 1.0, "dust_w": 0.0015}, **KW)
    assert banded.periods["turnover"].mean() < no_band.periods["turnover"].mean()
    assert banded.periods["n_banded"].sum() > 0


def test_conviction_changes_weights_but_not_gross(seeded):
    """Conviction tilts RELATIVE weights; the vol target still binds, so it must not
    quietly lever the book up."""
    flat = run_backtest(seeded, [UpProvider()], sizing=True, cfg={}, **KW)
    conv = run_backtest(seeded, [UpProvider()], sizing=True,
                        cfg={"conviction": True}, **KW)
    w_flat = list(flat.weights_history.values())[-1]
    w_conv = list(conv.weights_history.values())[-1]
    assert w_flat != w_conv
    assert sum(w_conv.values()) <= 1.0


def test_ranked_fn_is_a_free_shortcut(seeded):
    """The sweep memoises the ranking across configs. If that changed ANY number, every
    parameter chosen from the study would be chosen from the wrong backtest."""
    provs = [UpProvider()]

    memo: dict = {}

    def ranked_fn(as_of):
        if as_of not in memo:
            syms = list(seeded.universe_as_of(as_of).symbol)
            memo[as_of] = rank_universe(seeded, provs, syms, as_of)
        return memo[as_of]

    direct = run_backtest(seeded, provs, sizing=True, **KW)
    memoed = run_backtest(seeded, provs, sizing=True, ranked_fn=ranked_fn, **KW)
    pd.testing.assert_frame_equal(direct.periods, memoed.periods)


@pytest.mark.parametrize("freq,expected", [
    ("W-FRI", 52.0), ("2W-FRI", 26.0), ("4W-FRI", 13.0),
    ("ME", 12.0), ("M", 12.0), ("2M", 6.0), ("BME", 12.0),
    ("Q", 4.0), ("B", 252.0), ("D", 252.0), ("YE", 1.0),
])
def test_periods_per_year_honours_the_multiplier(freq, expected):
    """'2W-FRI' used to fall through every branch (it starts with '2', not 'W') and
    return 52.0 -- annualising a biweekly backtest as if it rebalanced weekly. That
    inflates Sharpe by sqrt(2) and roughly doubles CAGR, which is enough to make a
    cadence study 'prove' that trading less often earns more. It doesn't; the divisor
    was just wrong."""
    from qts.backtest.engine import _periods_per_year
    assert _periods_per_year(freq) == pytest.approx(expected)


def test_biweekly_is_not_annualised_as_weekly(seeded):
    res = run_backtest(seeded, [UpProvider()], sizing=True, freq="2W-FRI",
                       start=date(2025, 1, 1), end=date(2025, 6, 1), regime_on=False)
    assert res.periods_per_year == pytest.approx(26.0)
