"""The cost ledger exists because Alpaca paper charges NO crypto fees (measured: an implied
-3.3bps against a published 25bps taker rate). Paper P&L is therefore inflated by the full
round trip, and this ledger is the only honest NAV."""
import pytest

from qts.crypto.costs import CostLedger, CryptoCostModel


def test_maker_does_NOT_book_spread_capture():
    """THE most dangerous number in the whole project.

    An earlier draft modelled maker cost as `15 - half_spread/2`, which returned NEGATIVE
    costs on wide-spread coins -- i.e. it claimed the venue would PAY US to trade. Passive
    fills are signed against you (P(fill|adverse) ~= 0.99 vs P(fill|favourable) ~= 0.018),
    so the half-spread is handed straight back as adverse markout. Maker cost is the FEE.
    """
    c = CryptoCostModel()
    for hs in (0.0, 4.8, 19.1, 39.4, 200.0):
        assert c.maker_bps(hs) == 15.0, "maker cost must be the fee, with NO spread credit"
        assert c.maker_bps(hs) > 0, "cost may never be negative -- that is a free lunch"


def test_taker_crosses_the_spread():
    c = CryptoCostModel()
    assert c.taker_bps(4.8) == pytest.approx(29.8)     # BTC
    assert c.taker_bps(39.4) == pytest.approx(64.4)    # the alt tail
    assert c.taker_bps(0.0) == 25.0


def test_maker_is_always_cheaper_than_taker():
    c = CryptoCostModel()
    for hs in (0.0, 5.0, 20.0, 40.0):
        assert c.maker_bps(hs) < c.taker_bps(hs)


def test_adjusted_equity_subtracts_the_fees_the_broker_never_charged():
    led = CostLedger()
    led.record("BTC/USD", "buy", qty=1.0, fill_price=60_000, arrival_mid=60_000,
               half_spread_bps=4.8, liquidity="taker")
    # taker on BTC = 25 + 4.8 = 29.8bps of $60,000 = $178.80
    assert led.total_fee_usd() == pytest.approx(178.80, abs=0.01)
    assert led.adjusted_equity(100_000) == pytest.approx(100_000 - 178.80, abs=0.01)


def test_slippage_is_signed_against_the_side():
    led = CostLedger()
    buy = led.record("BTC/USD", "buy", 1.0, fill_price=60_060, arrival_mid=60_000,
                     half_spread_bps=5, liquidity="taker")
    sell = led.record("BTC/USD", "sell", 1.0, fill_price=59_940, arrival_mid=60_000,
                      half_spread_bps=5, liquidity="taker")
    assert buy.slippage_bps == pytest.approx(10.0)    # paid UP to buy -> positive cost
    assert sell.slippage_bps == pytest.approx(10.0)   # sold DOWN -> also a positive cost


def test_maker_rate_is_measured_not_assumed():
    """There is no post_only flag on Alpaca, so a resting order that crosses silently pays
    taker. The maker rate must come from what was actually charged."""
    led = CostLedger()
    led.record("BTC/USD", "buy", 1.0, 60_000, 60_000, 4.8, liquidity="maker")
    led.record("BTC/USD", "buy", 1.0, 60_000, 60_000, 4.8, liquidity="taker")
    assert led.maker_rate() == pytest.approx(0.5)
    s = led.summary()
    assert s["n_fills"] == 2 and s["fee_bps_effective"] == pytest.approx((15 + 29.8) / 2, abs=0.1)
