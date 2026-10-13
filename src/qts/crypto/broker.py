"""Crypto broker: maker-first execution, and an honest accounting of what that costs.

WHY MAKER-FIRST IS EXISTENTIAL, NOT AN OPTIMISATION.
Taker cost = 25bps fee + the half-spread you cross = 30bps on BTC, up to 64bps on the alt
tail. Maker cost = 15bps fee. That 10bps differential is the ENTIRE economic case, and it is
real and certain.

WHAT IS *NOT* REAL: SPREAD CAPTURE. Resting at the bid does not earn you half the spread.
Passive fills are signed against you -- P(fill | price moved against you) ~= 0.99 versus
P(fill | it moved your way) ~= 0.018 (DeLise, arXiv 2407.16527). You capture the half-spread
and hand ~90% of it back as adverse markout. See costs.py: we model that term as ZERO.

The strategy survives anyway because the alpha horizon here is MONTHS (a trend ensemble that
switches ~15-80 times a year) while the adverse-selection horizon is SECONDS. Patience is
free for us and ruinous for an HFT. That is the argument -- not spread capture.

NO post_only FLAG EXISTS ON ALPACA. Maker status can never be guaranteed: a resting order
that crosses silently pays 25bps instead of 15. So we RECONCILE the fee actually charged
rather than assuming, and `CostLedger.maker_rate()` measures what we really got.

VENUE FACTS, ALL VERIFIED AGAINST THE LIVE API 2026-07-14:
  - Order symbol is `BTC/USD`; the POSITION comes back as `BTCUSD`, and DELETE
    /v2/positions/{sym} only accepts the slashless form. Normalise at every boundary.
  - NOTIONAL crypto orders under-fill by ~2% (Alpaca reserves a price buffer): a $10 notional
    buy delivered $9.8036 of BTC. USE QTY ORDERS. This module never sends notional.
  - Crypto TIF is `gtc` / `ioc` only -- no `day`.
  - Paper charges NO crypto fees, so paper P&L is inflated by the full round trip.
  - 200 requests/min account-wide, which caps how fast we may re-price.
"""
from __future__ import annotations

import os
import time
from decimal import ROUND_DOWN, Decimal

import requests

from .costs import CostLedger, CryptoCostModel
from .prices import latest_quote
from .universe import from_venue, to_venue

PAPER = "https://paper-api.alpaca.markets/v2"

# Repricing cadence. 200 req/min account-wide is the hard ceiling; a chase loop faster than
# this earns 429s and, at retail latency with no colocation, buys nothing but queue churn.
REPRICE_SECONDS = 30
# How long to stay passive before crossing. Our alpha does not decay in an hour; if it did,
# passive execution would be the wrong tool entirely and we should be crossing from the start.
PASSIVE_TIMEOUT_S = 3600
# Hard cap on how far a chase may drift from the arrival mid. A chase without this is just a
# slow-motion market order.
MAX_CHASE_BPS = 30.0


# Alpaca crypto quantities carry NINE decimals (min_trade_increment 1e-9 on BTC/USD and
# ETH/USD, verified 2026-09-03). Positions come back with all nine. An order quantity must
# therefore be expressed to nine places and ROUNDED DOWN: formatting a 9-place position to
# eight places with round-half-up asked to sell 10.27529683 ETH out of 10.275296827 held,
# and the venue answered "403 insufficient balance for ETH". Every full exit whose ninth
# digit was >= 5 failed that way -- 14 of 21 scheduled rebalances between 07-23 and 09-02,
# including three straight weeks in which the trend rule tried to leave BTC and could not.
QTY_DECIMALS = 9
_QTY_QUANTUM = Decimal(1).scaleb(-QTY_DECIMALS)


def format_qty(qty: float) -> str:
    """Venue quantity string: exactly nine decimals, rounded DOWN, never up.

    Rounding down can under-fill by at most one billionth of a coin (well under a cent);
    rounding up by the same amount is a rejected order. Trailing zeros are kept so the
    string is unambiguous in logs."""
    if qty < 0:
        raise ValueError(f"qty must be non-negative, got {qty}")
    d = Decimal(repr(float(qty))).quantize(_QTY_QUANTUM, rounding=ROUND_DOWN)
    return f"{d:.{QTY_DECIMALS}f}"


class VenueRejected(RuntimeError):
    """A non-2xx from the trading API, WITH the venue's explanation attached.

    `requests.raise_for_status()` only carries the status line; the body is where Alpaca
    says why ("insufficient balance for ETH (requested: ..., available: ...)"). Losing it
    is what let the rounding bug above run for six weeks as an anonymous "403 Forbidden"."""

    def __init__(self, method: str, path: str, status: int, body: str, payload: dict | None = None):
        self.method, self.path, self.status, self.body, self.payload = method, path, status, body, payload
        super().__init__(f"{method} {path} -> {status}: {body[:400]}"
                         + (f" | sent {payload}" if payload else ""))


# Terminal order states -- a cancel is only DONE when the venue reports one of these.
_TERMINAL = {"canceled", "filled", "expired", "rejected", "replaced", "done_for_day"}
# How long to wait for a cancel to land before re-ordering. A DELETE returns 204 while
# the order is still `pending_cancel` and still holds its balance reservation; placing
# the replacement in that window is a second route to "insufficient balance".
CANCEL_SETTLE_S = 5.0


class CryptoBroker:
    """Alpaca crypto, maker-first. Requires its OWN account (see profile.guard_account)."""

    def __init__(self, key_id: str | None = None, secret_key: str | None = None,
                 base: str = PAPER, cost_model: CryptoCostModel | None = None,
                 max_half_spread_bps: float = 50.0, ledger_path=None):
        self._key = key_id or os.environ.get("APCA_CRYPTO_KEY_ID")
        self._sec = secret_key or os.environ.get("APCA_CRYPTO_SECRET_KEY")
        self.base = base
        # ledger_path=None keeps the ledger in memory (tests, dry runs). The live runner
        # passes a file so modelled fees ACCUMULATE across runs -- a per-run ledger made
        # `adjusted_equity` forget every fee the moment the process exited.
        self.ledger = CostLedger(cost_model or CryptoCostModel(), path=ledger_path)
        self.max_half_spread_bps = max_half_spread_bps

    # ---- plumbing -------------------------------------------------------
    @property
    def _h(self) -> dict:
        if not (self._key and self._sec):
            raise RuntimeError("crypto broker has no credentials "
                               "(APCA_CRYPTO_KEY_ID / APCA_CRYPTO_SECRET_KEY)")
        return {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._sec}

    def _get(self, path: str, **kw):
        r = requests.get(f"{self.base}{path}", headers=self._h, timeout=30, **kw)
        r.raise_for_status()
        return r.json()

    def equity(self) -> float:
        return float(self._get("/account")["equity"])

    def get_positions(self) -> dict[str, float]:
        """CANONICAL symbol -> weight. Crypto only: an equity position here would mean the
        guard failed and we are pointed at the wrong account, so we refuse rather than
        silently price it into the book."""
        eq = self.equity()
        if eq <= 0:
            return {}
        out: dict[str, float] = {}
        for p in self._get("/positions"):
            if p.get("asset_class") != "crypto":
                raise RuntimeError(
                    f"non-crypto position {p['symbol']} in the crypto account -- this book "
                    "is pointed at the equity account. Refusing to trade."
                )
            out[from_venue(p["symbol"])] = float(p["market_value"]) / eq
        return out

    def get_qty(self, symbol: str) -> float:
        for p in self._get("/positions"):
            if from_venue(p["symbol"]) == symbol:
                return float(p["qty"])
        return 0.0

    def qty_available(self, symbol: str) -> float:
        """Quantity the venue will actually let us sell right now (open orders reserve
        the rest). This, not `qty`, bounds a sell order."""
        for p in self._get("/positions"):
            if from_venue(p["symbol"]) == symbol:
                return float(p.get("qty_available", p["qty"]))
        return 0.0

    # ---- circuit breaker ------------------------------------------------
    def tradeable(self, symbol: str) -> tuple[bool, dict]:
        """Oct-10-2025 insurance. On that day depth fell >90% and spreads went from single
        -digit bps to double-digit PERCENT. If the book is that bad we SKIP the rebalance
        rather than trade through it -- a missed rebalance costs a day of drift; a rebalance
        into a vacuum costs real money."""
        q = latest_quote(symbol)
        ok = q["half_spread_bps"] <= self.max_half_spread_bps
        return ok, q

    # ---- execution ------------------------------------------------------
    def execute(self, symbol: str, delta_qty: float, dry_run: bool = True,
                passive_timeout_s: int = PASSIVE_TIMEOUT_S) -> dict:
        """Move `delta_qty` of `symbol`. Passive at the touch, then escalate to IOC.

        Phase 1 (passive): rest a limit AT THE TOUCH -- never inside it. Improving the touch
        on a wide book makes you the cheapest option for anyone with information, which is
        the textbook way to volunteer for adverse selection.
        Phase 2 (escalate): after the timeout, cross with a marketable limit + IOC, capped at
        arrival_mid +/- MAX_CHASE_BPS. Never a bare market order: a market order into a thin
        crypto book with no price bound is how you eat a 200bp fill.
        Residual is CARRIED to the next cycle and logged as opportunity cost, not chased.
        """
        side = "buy" if delta_qty > 0 else "sell"
        want = abs(delta_qty)
        ok, q = self.tradeable(symbol)
        arrival_mid, hs = q["mid"], q["half_spread_bps"]
        plan = {"symbol": symbol, "side": side, "qty": want, "arrival_mid": arrival_mid,
                "half_spread_bps": round(hs, 2), "tradeable": ok}
        if not ok:
            plan["status"] = "skipped_wide_spread"
            return plan
        if dry_run:
            plan["status"] = "dry_run"
            return plan

        # PARTIAL-FILL ACCOUNTING. The first live run (2026-07-19) proved why this is not
        # optional: a passive BTC order filled 0.0024 of 0.1286, was canceled on a reprice,
        # and the escalation then re-sent the FULL 0.1286 -- a $133 over-buy the code
        # reported as a clean "filled". Every order (working, canceled, or IOC) may carry
        # partial fills, so we book each increment exactly once and always place the NEXT
        # order for the REMAINING quantity, never the original want.
        booked: dict[str, float] = {}          # order_id -> qty already booked to the ledger

        def sync(state: dict, liquidity: str) -> None:
            oid = state.get("id")
            if not oid:
                return
            fq = float(state.get("filled_qty") or 0.0)
            fp = float(state.get("filled_avg_price") or 0.0)
            inc = fq - booked.get(oid, 0.0)
            if inc > 1e-12 and fp > 0:
                self.ledger.record(symbol, side, inc, fp, arrival_mid, hs, liquidity)
                booked[oid] = fq

        def filled() -> float:
            return sum(booked.values())

        dust = max(want * 1e-4, 1e-9)          # ignore sub-0.01% residuals as done

        touch = q["bid"] if side == "buy" else q["ask"]
        o = self._order(symbol, side, want, limit=touch, tif="gtc")
        plan["passive_order_id"] = o.get("id")

        deadline = time.time() + passive_timeout_s
        while time.time() < deadline:
            time.sleep(REPRICE_SECONDS)
            st = self._get(f"/orders/{o['id']}")
            sync(st, "maker")
            if st.get("status") == "filled" or want - filled() <= dust:
                return self._finish(plan, want, filled(), hs, escalated=False)
            nq = latest_quote(symbol)
            new_touch = nq["bid"] if side == "buy" else nq["ask"]
            drift = abs(new_touch / arrival_mid - 1) * 1e4
            if drift > MAX_CHASE_BPS:
                break                                  # chase cap: stop following the market
            if abs(new_touch - touch) / max(touch, 1e-9) > 1e-4:
                sync(self._cancel(o["id"]), "maker")   # capture fills BEFORE the cancel landed
                remaining = want - filled()
                if remaining <= dust:
                    return self._finish(plan, want, filled(), hs, escalated=False)
                touch = new_touch                      # one live order per symbol, always
                o = self._order(symbol, side, remaining, limit=touch, tif="gtc")
                plan["passive_order_id"] = o.get("id")

        # Escalate the REMAINING quantity only: marketable limit + IOC, price-bounded.
        sync(self._cancel(o["id"]), "maker")
        remaining = want - filled()
        plan["escalated"] = True
        if remaining <= dust:
            return self._finish(plan, want, filled(), hs, escalated=True)
        cap = arrival_mid * (1 + (MAX_CHASE_BPS / 1e4) * (1 if side == "buy" else -1))
        o2 = self._order(symbol, side, remaining, limit=cap, tif="ioc")
        time.sleep(3)
        sync(self._get(f"/orders/{o2['id']}"), "taker")
        return self._finish(plan, want, filled(), hs, escalated=True)

    def _finish(self, plan: dict, want: float, filled: float, hs: float,
                escalated: bool = False) -> dict:
        """Attach the outcome. All fills are already booked to the ledger by `sync`; this
        only classifies the result so the audit record is honest about a partial carry."""
        plan["escalated"] = plan.get("escalated", escalated)
        plan["filled_qty"] = round(filled, 10)
        plan["unfilled_qty"] = round(max(0.0, want - filled), 10)
        if filled >= want * (1 - 1e-4):
            plan["status"] = "filled"
        elif filled > 0:
            plan["status"] = "partial_carried"         # residual carried, not chased
        else:
            plan["status"] = "unfilled_carried"
        recent = [f for f in self.ledger.fills if f.symbol == plan["symbol"]]
        if recent:
            makers = [f for f in recent if f.liquidity == "maker"]
            plan["maker_qty"] = round(sum(f.qty for f in makers), 10)
            plan["avg_fill_price"] = round(
                sum(f.fill_price * f.qty for f in recent) / sum(f.qty for f in recent), 2)
        return plan

    def _order(self, symbol: str, side: str, qty: float, limit: float, tif: str) -> dict:
        # QTY, never notional: notional crypto orders under-fill by ~2% (measured).
        if side == "sell":
            # Bound by what the venue says is sellable. A full exit computed from the
            # position's own qty can still exceed qty_available by a rounding hair or by
            # a reservation held by an order in its cancel window -- either is a 403.
            avail = self.qty_available(symbol)
            if qty > avail:
                qty = avail
        body = {"symbol": symbol, "qty": format_qty(qty), "side": side, "type": "limit",
                "limit_price": f"{limit:.8f}", "time_in_force": tif}
        r = requests.post(f"{self.base}/orders", headers=self._h, json=body, timeout=30)
        if not r.ok:
            raise VenueRejected("POST", "/orders", r.status_code, r.text, body)
        return r.json()

    def _cancel(self, order_id: str) -> dict:
        """Cancel, then WAIT for the venue to confirm it, and return the final state so
        the caller can book any partial fill that landed before the cancel took effect.

        A cancel is not instantaneous: DELETE returns 204 while the order is still
        `pending_cancel`, and in that window it (a) can still fill -- exactly how the
        first live run over-bought -- and (b) still holds its balance reservation, so a
        replacement order placed too early is refused. Poll until a terminal status."""
        requests.delete(f"{self.base}/orders/{order_id}", headers=self._h, timeout=20)
        deadline = time.time() + CANCEL_SETTLE_S
        st: dict = {}
        while True:
            time.sleep(0.5)
            try:
                st = self._get(f"/orders/{order_id}")
            except Exception:  # noqa: BLE001  a cancelled order that 404s simply has no more to book
                return st or {}
            if st.get("status") in _TERMINAL or time.time() >= deadline:
                return st

    def close_all(self) -> list[dict]:
        """Kill-switch path. Uses the SLASHLESS symbol -- DELETE /positions/BTC/USD 404s."""
        out = []
        for p in self._get("/positions"):
            r = requests.delete(f"{self.base}/positions/{to_venue(from_venue(p['symbol']))}",
                                headers=self._h, timeout=30)
            out.append({"symbol": p["symbol"], "http": r.status_code})
        return out
