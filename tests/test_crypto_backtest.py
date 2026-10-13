"""The backtest must simulate the book we actually trade, and must not flatter it."""
import numpy as np
import pandas as pd
import pytest

from qts.crypto.backtest import run, stats
from qts.crypto.costs import CryptoCostModel
from qts.crypto import rebalance as rb
from qts.risk import trade_policy as tp


def _panel(n=900, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "BTC/USD": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.03, n))),
        "ETH/USD": 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.04, n))),
    }, index=idx)


def test_live_and_simulated_paths_bind_the_SAME_trade_policy():
    """Function identity, not 'equivalent logic'. This is the bug class that made the equity
    backtest simulate a portfolio nobody owned: an inline reimplementation on one side
    silently drifts from the other."""
    import qts.crypto.backtest as bt
    assert bt.apply_trade_policy is tp.apply_trade_policy
    assert rb.apply_trade_policy is tp.apply_trade_policy


def test_costs_strictly_reduce_returns():
    p = _panel()
    r = run(p, CryptoCostModel(), liquidity="taker")
    assert r["ann_cost_pct"] > 0
    assert r["net"]["cagr"] < r["gross"]["cagr"]


def test_taker_is_strictly_more_expensive_than_maker():
    p = _panel()
    mk = run(p, CryptoCostModel(), liquidity="maker")
    tk = run(p, CryptoCostModel(), liquidity="taker")
    assert tk["ann_cost_pct"] > mk["ann_cost_pct"]


def test_a_wider_band_cuts_turnover():
    p = _panel()
    tight = run(p, CryptoCostModel(), band_w=0.0)
    wide = run(p, CryptoCostModel(), band_w=0.08)
    assert wide["turnover_per_day_pct"] < tight["turnover_per_day_pct"]
    assert wide["ann_cost_pct"] < tight["ann_cost_pct"]


def test_book_is_never_levered():
    p = _panel()
    r = run(p, CryptoCostModel(), vol_target=5.0)   # absurd target: the cap must still bind
    assert r["periods"]["gross"].max() <= 1.0 + 1e-9


def test_sharpe_se_is_reported_because_the_sample_is_short():
    """5.5y of daily bars gives SE(Sharpe) ~ 0.45. Reporting a Sharpe without it, on a
    sample where the single-trial significance threshold is 0.84, is how you fool yourself."""
    s = stats(pd.Series(np.random.default_rng(0).normal(0.001, 0.03, 2000)))
    assert "sharpe_se" in s and 0.2 < s["sharpe_se"] < 1.0


def test_lower_vol_target_cannot_raise_sharpe_much_it_only_holds_less():
    """The trap this pins: a book that sits in cash posts a flattering DRAWDOWN while its
    Sharpe barely moves. Anyone selling the -10% drawdown of a 10%-vol-target book is
    selling '90% cash'. Sharpe is scale-invariant, so it is the honest comparator."""
    p = _panel()
    lo = run(p, CryptoCostModel(), vol_target=0.10)
    hi = run(p, CryptoCostModel(), vol_target=0.40)
    assert lo["avg_gross"] < hi["avg_gross"]
    assert abs(lo["net"]["max_dd"]) < abs(hi["net"]["max_dd"])   # drawdown "improves"...
    assert abs(lo["net"]["sharpe"] - hi["net"]["sharpe"]) < 0.35  # ...but Sharpe barely moves
