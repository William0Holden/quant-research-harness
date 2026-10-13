"""tests/test_daily_cadence.py"""
from qts.backtest.engine import _periods_per_year
from qts.exec.reconcile import compute_orders


def test_band_skips_small_drifts():
    cur = {"AAA": 0.100, "BBB": 0.050}
    tgt = {"AAA": 0.101, "BBB": 0.000}
    orders = compute_orders(cur, tgt, 100_000.0, band_w=0.002)
    assert [o["symbol"] for o in orders] == ["BBB"]        # AAA drift 10bps < 20bps band
    # band 0 (weekly default) keeps both
    assert len(compute_orders(cur, tgt, 100_000.0)) == 2


def test_band_never_blocks_full_liquidation_to_zero():
    orders = compute_orders({"AAA": 0.001}, {}, 100_000.0,
                            min_trade_notional=0.0, band_w=0.002)
    assert [o["symbol"] for o in orders] == ["AAA"]        # exit-to-zero always allowed


def test_periods_per_year_daily():
    assert _periods_per_year("B") == 252.0
    assert _periods_per_year("D") == 252.0
    assert _periods_per_year("W-FRI") == 52.0
    assert _periods_per_year("BME") == 12.0
