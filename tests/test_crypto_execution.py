"""Partial-fill accounting in the maker ladder.

Reproduces the bug the FIRST LIVE RUN exposed (2026-07-19): a passive BTC order filled
0.0024 of 0.1286, was canceled on a reprice, and the escalation re-sent the FULL 0.1286 --
a $133 over-buy the code reported as a clean 'filled', and which never touched the ledger.

The fake scripts _order / _get / _cancel / latest_quote / tradeable so the real execute()
orchestration runs with no network. The invariant under test: TOTAL FILLED NEVER EXCEEDS
WANT, and every partial is booked to the ledger exactly once.
"""
import pytest

from qts.crypto.broker import CryptoBroker


class _FakeVenue:
    """A scripted order book. Each _order() returns an id; the script says how much each
    order fills. Repriced/canceled orders can carry partial fills, like the real venue."""

    def __init__(self, fills_by_seq):
        # fills_by_seq[i] = (filled_qty, status) for the i-th order placed
        self.fills = fills_by_seq
        self.orders = []            # (id, requested_qty, tif)
        self.state = {}             # id -> current state dict

    def order(self, symbol, side, qty, limit, tif):
        oid = f"o{len(self.orders)}"
        fq, status = self.fills[len(self.orders)]
        fq = min(fq, qty)           # can never fill more than was requested
        self.orders.append((oid, qty, tif))
        self.state[oid] = {"id": oid, "filled_qty": str(fq), "status": status,
                           "filled_avg_price": "60000" if fq > 0 else None}
        return {"id": oid}

    def get(self, path):
        return self.state[path.split("/")[-1]]

    def cancel(self, oid):
        st = dict(self.state[oid])
        st["status"] = "canceled"
        return st


def _wire(monkeypatch, venue):
    b = CryptoBroker(key_id="k", secret_key="s")
    monkeypatch.setattr(b, "tradeable",
                        lambda s: (True, {"mid": 60000.0, "bid": 59994.0, "ask": 60006.0,
                                          "half_spread_bps": 10.0}))
    monkeypatch.setattr(b, "_order",
                        lambda symbol, side, qty, limit, tif: venue.order(symbol, side, qty, limit, tif))
    monkeypatch.setattr(b, "_get", lambda path, **kw: venue.get(path))
    monkeypatch.setattr(b, "_cancel", lambda oid: venue.cancel(oid))
    import qts.crypto.broker as mod
    monkeypatch.setattr(mod, "latest_quote",
                        lambda s: {"mid": 60000.0, "bid": 59994.0, "ask": 60006.0,
                                   "half_spread_bps": 10.0})
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    return b


def test_partial_then_escalate_does_NOT_overbuy(monkeypatch):
    """THE regression. Passive order partially fills 0.0024, never completes, escalation
    must send only want-0.0024 and total must equal want exactly -- not want+0.0024."""
    want = 0.1286
    # order 0: passive, partially fills 0.0024 then we (will) escalate. status stays working
    #          so the loop reprices/escalates. order 1: the IOC for the remainder.
    venue = _FakeVenue([(0.0024, "partially_filled"), (want - 0.0024, "filled")])
    b = _wire(monkeypatch, venue)

    res = b.execute("BTC/USD", want, dry_run=False, passive_timeout_s=0)  # 0 -> escalate at once

    total = sum(f.qty for f in b.ledger.fills)
    assert total == pytest.approx(want, abs=1e-9), f"over/under-bought: {total} vs {want}"
    # the escalation order must have requested only the REMAINDER, not the full want
    ioc_qty = venue.orders[-1][1]
    assert ioc_qty == pytest.approx(want - 0.0024, abs=1e-9)
    assert res["status"] == "filled"
    assert res["filled_qty"] == pytest.approx(want, abs=1e-9)


def test_every_partial_is_booked_exactly_once(monkeypatch):
    """The ledger must see BOTH the passive partial and the IOC fill -- the old code booked
    only the final IOC, so the cost ledger (our only honest NAV) under-counted."""
    want = 0.10
    venue = _FakeVenue([(0.03, "partially_filled"), (0.07, "filled")])
    b = _wire(monkeypatch, venue)
    b.execute("BTC/USD", want, dry_run=False, passive_timeout_s=0)

    qtys = sorted(round(f.qty, 6) for f in b.ledger.fills)
    assert qtys == [0.03, 0.07], f"expected both fills booked, got {qtys}"
    assert len(b.ledger.fills) == 2
    # first is maker (passive), second is taker (escalation)
    assert b.ledger.fills[0].liquidity == "maker"
    assert b.ledger.fills[1].liquidity == "taker"


def test_clean_passive_fill_never_escalates(monkeypatch):
    """The ETH case from the live run: a passive order that fills in full must not escalate,
    and must book exactly one maker fill."""
    want = 3.789
    venue = _FakeVenue([(want, "filled")])
    b = _wire(monkeypatch, venue)
    res = b.execute("ETH/USD", want, dry_run=False, passive_timeout_s=120)
    assert res["status"] == "filled"
    assert not res.get("escalated")
    assert len(b.ledger.fills) == 1 and b.ledger.fills[0].liquidity == "maker"
    assert len(venue.orders) == 1                      # no second order was ever placed


def test_total_unfilled_is_carried_not_rebought(monkeypatch):
    """If nothing fills, status is unfilled_carried and NOTHING is booked -- the residual
    goes to the next cycle, it is not chased."""
    want = 0.05
    venue = _FakeVenue([(0.0, "new"), (0.0, "canceled")])  # passive dead, IOC dead
    b = _wire(monkeypatch, venue)
    res = b.execute("BTC/USD", want, dry_run=False, passive_timeout_s=0)
    assert res["status"] == "unfilled_carried"
    assert res["unfilled_qty"] == pytest.approx(want)
    assert b.ledger.fills == []
