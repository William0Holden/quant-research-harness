"""Cost basis derived from fills.

The bug this guards: Alpaca paper reported HON at avg_entry_price=$49.95 against a
$247.90 mark (+$508.87 phantom gain) when the fills say $228.59 (+$49.65). With that
row corrupted the book's reported unrealized P&L had the wrong SIGN.
"""
from __future__ import annotations

import pytest

from qts.exec.cost_basis import (PositionBasis, average_cost_basis,
                                 fetch_fill_activities)


def _f(sym, side, qty, price, t):
    return {"symbol": sym, "side": side, "qty": str(qty), "price": str(price),
            "transaction_time": t}


# --------------------------------------------------------------------------- #
# the regression that started this
# --------------------------------------------------------------------------- #
def test_hon_basis_from_real_fills():
    """The two real HON fills, verbatim from the account's activity log."""
    fills = [
        _f("HON", "buy", "2", "229.61", "2026-07-06T13:34:05.012726Z"),
        _f("HON", "buy", "0.570711111", "225", "2026-07-20T13:30:00.741815Z"),
    ]
    hon = average_cost_basis(fills).positions["HON"]
    assert hon.qty == pytest.approx(2.570711111)
    assert hon.avg_cost == pytest.approx(228.5866, abs=1e-3)   # NOT the broker's 49.95
    assert hon.cost_basis == pytest.approx(587.63, abs=0.01)
    assert hon.unrealized_pl(247.90) == pytest.approx(49.65, abs=0.01)  # NOT 508.87


# --------------------------------------------------------------------------- #
# average-cost accounting
# --------------------------------------------------------------------------- #
def test_two_buys_average():
    r = average_cost_basis([
        _f("X", "buy", 10, 100, "2026-01-01T00:00:00Z"),
        _f("X", "buy", 10, 120, "2026-01-02T00:00:00Z"),
    ])
    p = r.positions["X"]
    assert p.qty == pytest.approx(20)
    assert p.avg_cost == pytest.approx(110)
    assert p.realized_pl == 0.0


def test_partial_sell_books_realized_and_keeps_avg():
    r = average_cost_basis([
        _f("X", "buy", 10, 100, "2026-01-01T00:00:00Z"),
        _f("X", "buy", 10, 120, "2026-01-02T00:00:00Z"),
        _f("X", "sell", 5, 130, "2026-01-03T00:00:00Z"),
    ])
    p = r.positions["X"]
    assert p.qty == pytest.approx(15)
    assert p.avg_cost == pytest.approx(110)          # average cost is unchanged by a sale
    assert p.realized_pl == pytest.approx(5 * (130 - 110))
    assert p.cost_basis == pytest.approx(15 * 110)


def test_full_exit_is_flat_with_no_residual_basis():
    r = average_cost_basis([
        _f("X", "buy", 3, 50, "2026-01-01T00:00:00Z"),
        _f("X", "sell", 3, 60, "2026-01-02T00:00:00Z"),
    ])
    p = r.positions["X"]
    assert p.qty == 0.0
    assert p.cost_basis == 0.0                        # no float dust left behind
    assert p.realized_pl == pytest.approx(30)
    assert p.unrealized_pl(999) is None               # flat has no unrealized P&L


def test_round_trip_then_reopen_starts_a_fresh_basis():
    r = average_cost_basis([
        _f("X", "buy", 1, 10, "2026-01-01T00:00:00Z"),
        _f("X", "sell", 1, 20, "2026-01-02T00:00:00Z"),
        _f("X", "buy", 1, 30, "2026-01-03T00:00:00Z"),
    ])
    p = r.positions["X"]
    assert p.avg_cost == pytest.approx(30)            # not blended with the closed lot
    assert p.realized_pl == pytest.approx(10)


def test_sell_through_zero_opens_a_short():
    r = average_cost_basis([
        _f("X", "buy", 5, 100, "2026-01-01T00:00:00Z"),
        _f("X", "sell", 8, 110, "2026-01-02T00:00:00Z"),
    ])
    p = r.positions["X"]
    assert p.qty == pytest.approx(-3)
    assert p.avg_cost == pytest.approx(110)           # the short is opened at the sale price
    assert p.realized_pl == pytest.approx(5 * (110 - 100))
    # A short profits when the price FALLS.
    assert p.unrealized_pl(100) == pytest.approx(30)


def test_short_cover_realizes_correctly():
    r = average_cost_basis([
        _f("X", "sell", 4, 50, "2026-01-01T00:00:00Z"),
        _f("X", "buy", 4, 40, "2026-01-02T00:00:00Z"),
    ])
    p = r.positions["X"]
    assert p.qty == 0.0
    assert p.realized_pl == pytest.approx(40)         # covered 10 lower on 4 units


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_fills_are_sorted_before_replay():
    """Average-cost is order-dependent; a mis-sorted page must not change the answer."""
    ordered = [
        _f("X", "buy", 10, 100, "2026-01-01T00:00:00Z"),
        _f("X", "sell", 5, 130, "2026-01-02T00:00:00Z"),
        _f("X", "buy", 10, 120, "2026-01-03T00:00:00Z"),
    ]
    a = average_cost_basis(ordered).positions["X"]
    b = average_cost_basis(list(reversed(ordered))).positions["X"]
    assert a.qty == pytest.approx(b.qty)
    assert a.avg_cost == pytest.approx(b.avg_cost)
    assert a.realized_pl == pytest.approx(b.realized_pl)


def test_malformed_rows_are_skipped_not_fatal():
    r = average_cost_basis([
        {"symbol": "X", "side": "buy", "qty": None, "price": "10",
         "transaction_time": "2026-01-01T00:00:00Z"},
        {"symbol": None, "side": "buy", "qty": "1", "price": "10",
         "transaction_time": "2026-01-01T00:00:00Z"},
        _f("X", "buy", 0, 10, "2026-01-01T00:00:00Z"),      # zero qty
        _f("X", "buy", 2, 10, "2026-01-02T00:00:00Z"),      # the only good row
    ])
    assert r.n_fills == 1
    assert r.positions["X"].qty == pytest.approx(2)


def test_report_totals_realized_across_symbols():
    r = average_cost_basis([
        _f("A", "buy", 1, 10, "2026-01-01T00:00:00Z"),
        _f("A", "sell", 1, 15, "2026-01-02T00:00:00Z"),
        _f("B", "buy", 1, 10, "2026-01-01T00:00:00Z"),
        _f("B", "sell", 1, 8, "2026-01-02T00:00:00Z"),
    ])
    assert r.realized_pl == pytest.approx(3)
    assert r.n_fills == 4


def test_empty_input():
    r = average_cost_basis([])
    assert r.positions == {} and r.realized_pl == 0.0 and not r.truncated


def test_position_basis_unrealized_needs_a_price():
    p = PositionBasis(symbol="X", qty=2, cost_basis=20)
    assert p.unrealized_pl(None) is None
    assert p.unrealized_pl(15) == pytest.approx(10)


# --------------------------------------------------------------------------- #
# paging
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    """Serves `pages` in order, recording the page_token the caller sent."""

    def __init__(self, pages):
        self.pages = pages
        self.tokens = []
        self.n = 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.tokens.append((params or {}).get("page_token"))
        page = self.pages[self.n] if self.n < len(self.pages) else []
        self.n += 1
        return _FakeResp(page)


def test_paging_follows_the_last_id_and_stops_on_a_short_page():
    full = [dict(_f("X", "buy", 1, 10, f"2026-01-01T00:00:{i:02d}Z"), id=f"id{i}")
            for i in range(100)]
    pages = [full, [dict(_f("X", "buy", 1, 10, "2026-01-02T00:00:00Z"), id="last")]]
    sess = _FakeSession(pages)
    fills, truncated = fetch_fill_activities("k", "s", session=sess)
    assert len(fills) == 101
    assert truncated is False
    assert sess.tokens == [None, "id99"]      # second page resumed from the first's last id


def test_truncation_is_reported_not_silent(monkeypatch):
    import qts.exec.cost_basis as CB

    monkeypatch.setattr(CB, "MAX_PAGES", 2)
    page = [dict(_f("X", "buy", 1, 10, f"2026-01-01T00:00:{i:02d}Z"), id=f"id{i}")
            for i in range(100)]
    sess = _FakeSession([page, page, page])
    fills, truncated = fetch_fill_activities("k", "s", session=sess)
    assert len(fills) == 200
    assert truncated is True                  # a capped basis must announce itself


def test_missing_keys_raise(monkeypatch):
    # Explicitly unset: the function falls back to env, so a dev box with real keys
    # exported would otherwise make this pass for the wrong reason.
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="keys are not set"):
        fetch_fill_activities("", "", session=_FakeSession([]))
