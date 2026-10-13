"""The crypto universe, and the symbol-normalisation footgun.

BTC and ETH only. This is not laziness -- it is the diversification arithmetic. At the
measured mean pairwise correlation of 0.707, going from 5 coins to 20 reduces book
volatility by ~3% while quadrupling transaction costs, operational surface and
idiosyncratic blow-up risk (delisting, token unlock, rug). A 20-coin crypto book contains
about 1.4-3.5 INDEPENDENT bets, not 20. Breadth is an illusion here.

BTC/USD and ETH/USD are also the only two pairs on the venue with (a) tight spreads --
4.8bps and 6.7bps half-spread vs 19-39bps for the alts -- and (b) full history back to
2021-01-01. Both matter more than coin count.

SYMBOL NORMALISATION (verified against the live API 2026-07-14): you SUBMIT an order for
`BTC/USD`, but the position comes back from /v2/positions as `BTCUSD`, and DELETE
/v2/positions/{sym} only accepts the slashless form. Getting this wrong means a position
silently fails to match its target and the book quietly drifts. Hence one canonical form
(the slashed pair) plus explicit converters at every venue boundary.
"""
from __future__ import annotations

# Canonical form used everywhere inside QTS. Order API accepts this form.
BTC = "BTC/USD"
ETH = "ETH/USD"
UNIVERSE: tuple[str, ...] = (BTC, ETH)

# First tradable bar on the venue. A backtest must not invent history before this:
# aggregators backfill pre-listing prices from venues you could not have traded.
LISTED_FROM: dict[str, str] = {BTC: "2021-01-01", ETH: "2021-01-01"}

# Measured live 2026-07-14. Used by the cost model; refreshed by ingest, not hardcoded
# into decisions. Half-spread in basis points.
HALF_SPREAD_BPS: dict[str, float] = {BTC: 4.8, ETH: 6.7}


def to_venue(symbol: str) -> str:
    """Canonical -> the form /v2/positions and DELETE /v2/positions/{sym} use."""
    return symbol.replace("/", "")


def from_venue(symbol: str) -> str:
    """Venue position symbol (BTCUSD) -> canonical (BTC/USD).

    Only USD-quoted pairs exist in this book, so the quote currency is unambiguous.
    Raises rather than guessing: a symbol we cannot map is a symbol we must not trade.
    """
    if "/" in symbol:
        return symbol
    if symbol.endswith("USD"):
        return f"{symbol[:-3]}/USD"
    raise ValueError(f"cannot map venue symbol {symbol!r} to a canonical pair")


def is_crypto(symbol: str) -> bool:
    """True for anything this book may hold, in EITHER symbol form.

    This is the guard the equity book uses to refuse to touch crypto positions, so it is a
    whitelist, not a heuristic. A pattern like `endswith("USD")` would be a landmine: it
    would silently classify any future equity ticker ending in those letters as crypto and
    exclude it from the equity book's targets -- which, since a symbol absent from the
    targets has an implied target weight of zero, is indistinguishable from "liquidate it".
    Guessing here is how you lose a position.
    """
    return symbol in _ALL_FORMS


_ALL_FORMS: frozenset[str] = frozenset(UNIVERSE) | {to_venue(s) for s in UNIVERSE}
