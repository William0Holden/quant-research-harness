"""The 2026-09-03 root cause: 14 of 21 live crypto rebalances died on an anonymous
"403 Forbidden". Alpaca positions carry nine decimals; the order path formatted eight with
round-half-up, so every full exit whose ninth digit was >= 5 asked to sell more than was
held and was refused -- and the refusal body was never logged. These tests pin all three
fixes: quantity formatting, sell clamping, and the surfaced venue explanation."""
import json

import pytest

from qts.crypto.broker import CryptoBroker, VenueRejected, format_qty
from qts.crypto.costs import CostLedger


# ---- quantity formatting ------------------------------------------------------------
@pytest.mark.parametrize("qty, expected", [
    (10.275296827, "10.275296827"),     # the live ETH position: nine places, kept exactly
    (0.526215335, "0.526215335"),
    (10.2752968279, "10.275296827"),    # a tenth decimal rounds DOWN, never up
    (0.1282385375, "0.128238537"),      # the July BTC position that could not be sold
    (0.3, "0.300000000"),
    (0.0, "0.000000000"),
])
def test_format_qty_is_nine_places_rounded_down(qty, expected):
    assert format_qty(qty) == expected


def test_format_qty_never_exceeds_input():
    # Property the venue enforces: requested <= available. Sample a grid of awkward floats.
    for i in range(1, 2000):
        q = i / 7.0 * 1e-3
        assert float(format_qty(q)) <= q + 1e-15


def test_format_qty_rejects_negative():
    with pytest.raises(ValueError):
        format_qty(-1e-9)


# ---- the order path -----------------------------------------------------------------
class _Resp:
    def __init__(self, status, body):
        self.status_code, self.ok, self._body = status, status < 400, body

    @property
    def text(self):
        return json.dumps(self._body)

    def json(self):
        return self._body


def _broker(monkeypatch, positions, post_status=200):
    b = CryptoBroker(key_id="k", secret_key="s")
    sent = []
    monkeypatch.setattr(b, "_get", lambda path, **kw: positions if path == "/positions" else {})

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        if post_status != 200:
            return _Resp(post_status, {"code": 40310000,
                                       "message": "insufficient balance for ETH "
                                                  "(requested: 10.27529683, available: 10.275296827)"})
        return _Resp(200, {"id": "o1", **json})

    import qts.crypto.broker as mod
    monkeypatch.setattr(mod.requests, "post", fake_post)
    return b, sent


def test_full_exit_sells_exactly_the_position(monkeypatch):
    pos = [{"symbol": "ETHUSD", "qty": "10.275296827", "qty_available": "10.275296827"}]
    b, sent = _broker(monkeypatch, pos)
    b._order("ETH/USD", "sell", 10.275296827, limit=2448.5, tif="gtc")
    assert sent[0]["qty"] == "10.275296827"          # was "10.27529683" -> 403


def test_sell_is_clamped_to_qty_available(monkeypatch):
    # An order still in its cancel window reserves part of the position.
    pos = [{"symbol": "BTCUSD", "qty": "0.526215335", "qty_available": "0.400000000"}]
    b, sent = _broker(monkeypatch, pos)
    b._order("BTC/USD", "sell", 0.526215335, limit=79000.0, tif="gtc")
    assert sent[0]["qty"] == "0.400000000"


def test_buy_is_not_clamped_by_position(monkeypatch):
    pos = [{"symbol": "BTCUSD", "qty": "0.1", "qty_available": "0.1"}]
    b, sent = _broker(monkeypatch, pos)
    b._order("BTC/USD", "buy", 0.25, limit=79000.0, tif="gtc")
    assert sent[0]["qty"] == "0.250000000"


def test_rejection_carries_the_venue_explanation(monkeypatch):
    pos = [{"symbol": "ETHUSD", "qty": "10.275296827", "qty_available": "10.275296827"}]
    b, _ = _broker(monkeypatch, pos, post_status=403)
    with pytest.raises(VenueRejected) as ei:
        b._order("ETH/USD", "sell", 1.0, limit=2448.5, tif="gtc")
    msg = str(ei.value)
    assert "403" in msg and "insufficient balance" in msg and "available: 10.275296827" in msg
    assert ei.value.payload["qty"] == "1.000000000"   # what we sent is in the record too


# ---- cancel waits for a terminal state --------------------------------------------
def test_cancel_polls_until_terminal(monkeypatch):
    import qts.crypto.broker as mod
    b = CryptoBroker(key_id="k", secret_key="s")
    states = iter([{"id": "o1", "status": "pending_cancel", "filled_qty": "0"},
                   {"id": "o1", "status": "pending_cancel", "filled_qty": "0.5"},
                   {"id": "o1", "status": "canceled", "filled_qty": "0.5"}])
    monkeypatch.setattr(mod.requests, "delete", lambda *a, **k: None)
    monkeypatch.setattr(b, "_get", lambda path, **kw: next(states))
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    st = b._cancel("o1")
    assert st["status"] == "canceled" and st["filled_qty"] == "0.5"


# ---- ledger persists across processes ----------------------------------------------
def test_ledger_accumulates_fees_across_runs(tmp_path):
    path = tmp_path / "fills.jsonl"
    run1 = CostLedger(path=path)
    run1.record("BTC/USD", "buy", 0.1, 60000.0, 60000.0, 2.0, "maker")
    fee1 = run1.total_fee_usd()
    assert fee1 == pytest.approx(0.1 * 60000 * 15 / 1e4)

    run2 = CostLedger(path=path)                      # a new process, same book
    assert run2.total_fee_usd() == pytest.approx(fee1)   # remembered
    run2.record("ETH/USD", "sell", 1.0, 2400.0, 2400.0, 1.0, "taker")
    assert run2.total_fee_usd() == pytest.approx(fee1 + 2400 * (25 + 1.0) / 1e4)   # taker = fee + half-spread
    assert run2.run_summary()["n_fills"] == 1 and run2.summary()["n_fills"] == 2
    assert run2.adjusted_equity(100_000.0) == pytest.approx(100_000.0 - run2.total_fee_usd())


def test_ledger_without_path_is_in_memory_only(tmp_path):
    led = CostLedger()
    led.record("BTC/USD", "buy", 0.1, 60000.0, 60000.0, 2.0, "maker")
    assert led.path is None and led.summary()["n_fills"] == 1 and not list(tmp_path.iterdir())
