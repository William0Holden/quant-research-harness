"""Tests for the execution layer — pure, no network or API keys required."""
from __future__ import annotations

import pytest

from qts.exec import DryRunBroker, TargetPortfolio, compute_orders


# --------------------------- compute_orders ---------------------------

def _by_symbol(orders):
    return {o["symbol"]: o for o in orders}


def test_buy_and_sell_diffs():
    current = {"AAPL": 0.5, "MSFT": 0.5}
    target = {"AAPL": 0.7, "MSFT": 0.3}
    orders = compute_orders(current, target, equity=10_000.0)
    by = _by_symbol(orders)

    assert by["AAPL"]["side"] == "buy"
    assert by["AAPL"]["notional"] == pytest.approx(2_000.0)
    assert by["AAPL"]["from_w"] == 0.5 and by["AAPL"]["to_w"] == 0.7

    assert by["MSFT"]["side"] == "sell"
    assert by["MSFT"]["notional"] == pytest.approx(2_000.0)


def test_min_notional_skip():
    current = {"AAPL": 0.50}
    target = {"AAPL": 0.5005}  # 0.0005 * 10_000 = $5 -> above default $1
    orders = compute_orders(current, target, equity=10_000.0)
    assert len(orders) == 1

    # Now make the move tiny relative to a higher floor -> skipped.
    orders2 = compute_orders(current, target, equity=10_000.0, min_trade_notional=10.0)
    assert orders2 == []


def test_min_notional_skip_small_equity():
    current = {"AAPL": 0.0}
    target = {"AAPL": 0.0001}  # 0.0001 * 100 = $0.01 -> below $1 floor
    orders = compute_orders(current, target, equity=100.0)
    assert orders == []


def test_full_liquidation():
    current = {"AAPL": 0.4, "MSFT": 0.6}
    target = {}  # liquidate everything
    orders = compute_orders(current, target, equity=5_000.0)
    by = _by_symbol(orders)
    assert by["AAPL"]["side"] == "sell" and by["AAPL"]["to_w"] == 0.0
    assert by["AAPL"]["notional"] == pytest.approx(2_000.0)
    assert by["MSFT"]["side"] == "sell" and by["MSFT"]["to_w"] == 0.0
    assert by["MSFT"]["notional"] == pytest.approx(3_000.0)


def test_new_position():
    current = {}  # holding only cash
    target = {"NVDA": 0.25}
    orders = compute_orders(current, target, equity=8_000.0)
    assert len(orders) == 1
    o = orders[0]
    assert o["symbol"] == "NVDA"
    assert o["side"] == "buy"
    assert o["from_w"] == 0.0 and o["to_w"] == 0.25
    assert o["notional"] == pytest.approx(2_000.0)
    assert o["dry_run"] is True


def test_no_change_no_orders():
    current = {"AAPL": 0.5, "MSFT": 0.5}
    orders = compute_orders(current, dict(current), equity=10_000.0)
    assert orders == []


def test_sells_before_buys_ordering():
    current = {"AAPL": 0.8, "ZZZ": 0.0}
    target = {"AAPL": 0.2, "ZZZ": 0.6}
    orders = compute_orders(current, target, equity=10_000.0)
    # First order should be the sell (AAPL), then the buy (ZZZ).
    assert orders[0]["symbol"] == "AAPL" and orders[0]["side"] == "sell"
    assert orders[1]["symbol"] == "ZZZ" and orders[1]["side"] == "buy"


def test_negative_equity_raises():
    with pytest.raises(ValueError):
        compute_orders({}, {"AAPL": 0.5}, equity=-1.0)


# --------------------------- DryRunBroker -----------------------------

def test_dryrun_broker_empty_positions():
    b = DryRunBroker()
    assert b.get_positions() == {}


def test_dryrun_broker_rebalance_orders():
    b = DryRunBroker()
    target = TargetPortfolio(weights={"AAPL": 0.6, "MSFT": 0.4})
    orders = b.rebalance(target)
    by = {o["symbol"]: o for o in orders}
    assert set(by) == {"AAPL", "MSFT"}
    # Same order-dict shape as the Alpaca path: {symbol, side, notional, from_w, to_w, dry_run}
    assert by["AAPL"]["side"] == "buy" and by["AAPL"]["from_w"] == 0.0 and by["AAPL"]["to_w"] == 0.6
    assert by["MSFT"]["side"] == "buy" and by["MSFT"]["from_w"] == 0.0 and by["MSFT"]["to_w"] == 0.4
    assert all(o["dry_run"] is True for o in orders)


def test_dryrun_broker_default_is_dry_run():
    # rebalance must default to dry_run=True (no venue contact in v1).
    b = DryRunBroker()
    target = TargetPortfolio(weights={"AAPL": 1.0})
    orders = b.rebalance(target)  # no dry_run arg passed
    assert orders and all(o["dry_run"] is True for o in orders)


# ------------------- AlpacaBroker live-loop order isolation -------------------
# Regression for 2026-07-03: one "not fractionable" APIError (HON) aborted the
# submission loop mid-rebalance, leaving ~90 orders on the book and the rest
# never sent. The live loop must isolate per-order failures and fall back to
# whole-share qty orders on non-fractionable assets.

from alpaca.common.exceptions import APIError  # noqa: E402

from qts.exec.broker import AlpacaBroker  # noqa: E402

NOT_FRACTIONABLE = '{"code":40310000,"message":"asset \\"HON\\" is not fractionable"}'


class FakeTradingClient:
    """Records submitted orders; raises per-symbol errors on first notional try."""

    def __init__(self, notional_rejects=(), hard_rejects=()):
        self.notional_rejects = set(notional_rejects)  # non-fractionable symbols
        self.hard_rejects = set(hard_rejects)          # always-fail symbols
        self.submitted = []

    def submit_order(self, req):
        if req.symbol in self.hard_rejects:
            raise APIError('{"code":40010001,"message":"forbidden"}')
        if req.symbol in self.notional_rejects and req.notional is not None:
            raise APIError(NOT_FRACTIONABLE.replace("HON", req.symbol))
        self.submitted.append(req)


def _live_broker(fake, prices=None, equity=10_000.0, positions=None):
    b = AlpacaBroker(key_id="k", secret_key="s")
    b._client = fake
    b.get_positions = lambda: dict(positions or {})
    b._account_equity = lambda: equity
    b._latest_trade_price = lambda sym: (prices or {}).get(sym, 100.0)
    return b


def test_alpaca_non_fractionable_falls_back_to_whole_shares():
    fake = FakeTradingClient(notional_rejects={"HON"})
    b = _live_broker(fake, prices={"HON": 200.0})
    out = b.rebalance(TargetPortfolio(weights={"HON": 0.05}), dry_run=False)
    o = out[0]
    assert o["status"] == "submitted" and o["qty"] == 2  # $500 // $200 = 2 shares
    assert fake.submitted[-1].qty == 2 and fake.submitted[-1].notional is None


def test_alpaca_non_fractionable_below_one_share_skipped():
    fake = FakeTradingClient(notional_rejects={"HON"})
    b = _live_broker(fake, prices={"HON": 900.0})
    out = b.rebalance(TargetPortfolio(weights={"HON": 0.05}), dry_run=False)
    assert out[0]["status"] == "skipped"
    assert fake.submitted == []  # nothing reached the venue


def test_alpaca_one_bad_order_does_not_strand_the_rest():
    fake = FakeTradingClient(hard_rejects={"BBB"})
    b = _live_broker(fake)
    out = b.rebalance(
        TargetPortfolio(weights={"AAA": 0.3, "BBB": 0.3, "CCC": 0.3}), dry_run=False)
    by = {o["symbol"]: o for o in out}
    assert by["BBB"]["status"] == "error" and "forbidden" in by["BBB"]["error"]
    assert by["AAA"]["status"] == "submitted" and by["CCC"]["status"] == "submitted"
    assert {r.symbol for r in fake.submitted} == {"AAA", "CCC"}
