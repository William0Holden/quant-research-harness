"""Cost basis derived from our own fill history, not from the broker's `avg_entry_price`.

WHY THIS EXISTS (2026-07-28). Alpaca paper reported HON at avg_entry_price=$49.95 against
a $247.90 mark -- a phantom +$508.87 (+396%) unrealized gain on a position whose actual
fills were 2 sh @ $229.61 (2026-07-06) and 0.5707 sh @ $225.00 (2026-07-20), i.e. a true
basis of $228.59 and a real gain of ~$50. With that one row corrupted, the book's reported
unrealized P&L (+$179) had the wrong SIGN: the truth is about -$280.

Account *equity* is unaffected -- it marks positions at live prices and never touches
basis -- but every per-position P&L number is only as good as the basis behind it. Fills
are the authoritative record of what we paid, so we recompute from them and keep the
broker's number visible beside ours wherever the two disagree.

The accounting is average-cost, signed, and handles a position crossing through zero, so
the same function serves the long-only equity book and anything short-capable later.

No Streamlit, no caching, no globals: `average_cost_basis` is a pure function over a list
of fill dicts and is unit-tested as such.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"

# Bound on paging. ~160 fills/week at the current weekly cadence, so 100 pages of 100 is
# roughly 12 years of history -- but if it is ever hit we say so rather than silently
# reporting a basis computed from a truncated tail (which would be wrong, not merely partial).
MAX_PAGES = 100
PAGE_SIZE = 100

_EPS = 1e-9


@dataclass
class PositionBasis:
    """Derived basis for one symbol. `qty` is signed (negative = short)."""
    symbol: str
    qty: float = 0.0
    cost_basis: float = 0.0      # signed: what the open position cost us
    realized_pl: float = 0.0
    n_fills: int = 0
    first_fill: str | None = None
    last_fill: str | None = None

    @property
    def avg_cost(self) -> float:
        return (self.cost_basis / self.qty) if abs(self.qty) > _EPS else 0.0

    def unrealized_pl(self, price: float | None) -> float | None:
        if price is None or abs(self.qty) <= _EPS:
            return None
        return self.qty * price - self.cost_basis

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_cost": self.avg_cost,
            "cost_basis": self.cost_basis,
            "realized_pl": self.realized_pl,
            "n_fills": self.n_fills,
            "first_fill": self.first_fill,
            "last_fill": self.last_fill,
        }


@dataclass
class BasisReport:
    positions: dict[str, PositionBasis] = field(default_factory=dict)
    n_fills: int = 0
    truncated: bool = False      # paging bound hit -> basis is NOT trustworthy, say so

    @property
    def realized_pl(self) -> float:
        return sum(p.realized_pl for p in self.positions.values())


def average_cost_basis(fills: list[dict]) -> BasisReport:
    """Replay fills chronologically into per-symbol average-cost basis.

    Each fill needs: symbol, side ('buy'/'sell'), qty, price, transaction_time.
    Rows missing a usable qty/price are skipped rather than poisoning the basis.

    Ordering is by transaction_time and NOT assumed from the caller -- average-cost
    accounting is order-dependent, so a mis-sorted page would silently produce a wrong
    basis rather than an error.
    """
    report = BasisReport()

    def _key(f: dict) -> str:
        return str(f.get("transaction_time") or "")

    for f in sorted(fills, key=_key):
        sym = f.get("symbol")
        if not sym:
            continue
        try:
            q = abs(float(f.get("qty")))
            p = float(f.get("price"))
        except (TypeError, ValueError):
            continue
        if q <= _EPS or p <= 0:
            continue

        side = str(f.get("side", "")).lower()
        signed = q if side == "buy" else -q
        pos = report.positions.setdefault(sym, PositionBasis(symbol=sym))
        ts = _key(f) or None

        if abs(pos.qty) <= _EPS or (pos.qty > 0) == (signed > 0):
            # Opening or adding in the same direction.
            pos.qty += signed
            pos.cost_basis += signed * p
        else:
            # Reducing, closing, or reversing.
            avg = pos.cost_basis / pos.qty
            direction = 1.0 if pos.qty > 0 else -1.0
            closed = min(q, abs(pos.qty))
            # Long closes profit when p > avg; short closes profit when p < avg.
            pos.realized_pl += direction * closed * (p - avg)
            pos.qty -= direction * closed
            pos.cost_basis -= direction * closed * avg
            remainder = q - closed
            if remainder > _EPS:                    # crossed through zero: open the other way
                new_dir = 1.0 if signed > 0 else -1.0
                pos.qty += new_dir * remainder
                pos.cost_basis += new_dir * remainder * p

        if abs(pos.qty) <= _EPS:                    # flat: clear float dust from the basis
            pos.qty = 0.0
            pos.cost_basis = 0.0

        pos.n_fills += 1
        if pos.first_fill is None:
            pos.first_fill = ts
        pos.last_fill = ts
        report.n_fills += 1

    return report


def fetch_fill_activities(
    key_id: str | None = None,
    secret_key: str | None = None,
    paper: bool = True,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> tuple[list[dict], bool]:
    """Page every FILL activity for the account. Returns (fills, truncated).

    alpaca-py exposes no activities client (checked against 0.43.4), so this speaks to
    the REST endpoint directly -- the same approach risk/regime.py takes for FRED.
    Raises on auth/network failure; callers that must degrade catch it.
    """
    key_id = key_id or os.environ.get("APCA_API_KEY_ID")
    secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
    if not (key_id and secret_key):
        raise RuntimeError("Alpaca API keys are not set (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")

    base = PAPER_BASE if paper else LIVE_BASE
    url = f"{base}/v2/account/activities/FILL"
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
    http = session or requests

    out: list[dict] = []
    token: str | None = None
    for _ in range(MAX_PAGES):
        params = {"page_size": PAGE_SIZE, "direction": "asc"}
        if token:
            params["page_token"] = token
        r = http.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            return out, False
        out.extend(batch)
        if len(batch) < PAGE_SIZE:
            return out, False
        token = batch[-1].get("id")
        if not token:
            return out, False
    return out, True


def account_basis(
    key_id: str | None = None,
    secret_key: str | None = None,
    paper: bool = True,
    session: requests.Session | None = None,
) -> BasisReport:
    """Convenience: fetch fills and reduce them to a BasisReport."""
    fills, truncated = fetch_fill_activities(key_id, secret_key, paper=paper, session=session)
    report = average_cost_basis(fills)
    report.truncated = truncated
    return report
