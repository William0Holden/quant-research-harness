"""Universe <-> venue symbol reconciliation for the equity book.

Aug 2026: every weekly rebalance errored on BRK-B ("not found" -- Alpaca spells it BRK.B)
and on EA / AVB / EQR ("not active" -- acquired or delisted, still in the universe). The
sizer kept handing those names weight (EA: 2.9% of target) that then sat in cash behind
a weekly order error. Fix: map class-share symbols at the boundary, and reconcile the
universe against the venue's active asset list BEFORE ranking."""
from types import SimpleNamespace

import pytest

from qts.exec.broker import AlpacaBroker, TargetPortfolio, from_venue, to_venue


def test_class_share_symbols_round_trip():
    assert to_venue("BRK-B") == "BRK.B" and from_venue("BRK.B") == "BRK-B"
    assert to_venue("BF-B") == "BF.B"
    assert to_venue("AAPL") == "AAPL" and from_venue("AAPL") == "AAPL"


class _Fake:
    def __init__(self, assets, positions=(), equity=10_000.0):
        self._assets = [SimpleNamespace(symbol=s, tradable=t) for s, t in assets]
        self._positions = [SimpleNamespace(symbol=s, market_value=mv, asset_class="us_equity")
                           for s, mv in positions]
        self._equity = equity
        self.calls = 0
        self.submitted = []

    def get_all_assets(self, req):
        self.calls += 1
        return self._assets

    def get_all_positions(self):
        return self._positions

    def get_account(self):
        return SimpleNamespace(equity=self._equity)

    def submit_order(self, req):
        self.submitted.append(req)

    def close_position(self, sym):
        self.submitted.append(SimpleNamespace(symbol=sym, notional=None, qty="all"))


def _broker(fake):
    b = AlpacaBroker(key_id="k", secret_key="s")
    b._client = fake
    return b


def test_filter_tradable_drops_names_the_venue_cannot_trade():
    fake = _Fake(assets=[("AAPL", True), ("BRK.B", True), ("HALT", False)])
    b = _broker(fake)
    ok, bad = b.filter_tradable(["AAPL", "BRK-B", "EA", "HALT"])
    assert ok == ["AAPL", "BRK-B"]              # BRK-B matched through the dot form
    assert set(bad) == {"EA", "HALT"}            # delisted, and listed-but-halted
    b.filter_tradable(["AAPL"])
    assert fake.calls == 1                       # asset list fetched once per broker


def test_positions_are_keyed_in_universe_form():
    fake = _Fake(assets=[], positions=[("BRK.B", 1_000.0), ("MSFT", 500.0)])
    b = _broker(fake)
    w = b.get_positions()
    assert set(w) == {"BRK-B", "MSFT"} and w["BRK-B"] == pytest.approx(0.1)


def test_orders_go_to_the_venue_in_venue_form_but_are_audited_in_universe_form():
    fake = _Fake(assets=[], positions=[("BRK.B", 1_000.0)])
    b = _broker(fake)
    b._latest_trade_price = lambda sym: 100.0
    out = b.rebalance(TargetPortfolio(weights={"BRK-B": 0.2, "BF-B": 0.05}), dry_run=False)
    assert {r.symbol for r in fake.submitted} == {"BRK.B", "BF.B"}
    assert {o["symbol"] for o in out} == {"BRK-B", "BF-B"}
    assert all(o["status"] == "submitted" for o in out)


def test_full_exit_closes_the_venue_symbol():
    fake = _Fake(assets=[], positions=[("BRK.B", 1_000.0)])
    b = _broker(fake)
    out = b.rebalance(TargetPortfolio(weights={}), dry_run=False)
    assert fake.submitted[0].symbol == "BRK.B" and out[0]["qty"] == "all"
