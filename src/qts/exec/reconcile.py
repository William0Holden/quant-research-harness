"""Pure order-reconciliation core (no venue, no keys, fully unit-testable).

`compute_orders` diffs current vs target portfolio weights into a list of
notional buy/sell orders. AlpacaBroker.rebalance delegates here so the trading
math can be tested without any network, API keys, or the alpaca-py package.
"""
from __future__ import annotations


def compute_orders(
    current: dict[str, float],
    target: dict[str, float],
    equity: float,
    min_trade_notional: float = 1.0,
    band_w: float = 0.0,
) -> list[dict]:
    """Diff current vs target weights into notional buy/sell orders.

    Args:
        current: symbol -> current weight (market value / equity).
        target: symbol -> target weight (long-only; sum <= 1).
        equity: account equity in account currency; notional = |dw| * equity.
        min_trade_notional: skip orders whose notional is below this (in $).
        band_w: no-trade band (weight units, e.g. 0.002 = 20bps of equity) for
            daily-cadence turnover control; drifts with |dw| < band_w are
            skipped EXCEPT a full exit to exactly 0, which always trades.

    Returns:
        A list of order dicts, one per symbol that needs to move, each:
            {symbol, side, notional, from_w, to_w, dry_run}
        side is "buy" (weight increasing) or "sell" (weight decreasing,
        including full liquidation to 0). dry_run is always True here — it is
        a pure description of intended orders; the broker decides submission.
        Orders are sorted sells-first then by symbol for deterministic output.
    """
    if equity < 0:
        raise ValueError(f"equity must be non-negative, got {equity}")

    symbols = set(current) | set(target)
    orders: list[dict] = []
    for sym in symbols:
        from_w = float(current.get(sym, 0.0))
        to_w = float(target.get(sym, 0.0))
        dw = to_w - from_w
        # No-trade band (daily cadence turnover control): skip drifts smaller than
        # band_w, EXCEPT full exits — a position going to exactly 0 always trades.
        if band_w and abs(dw) < band_w and not (to_w == 0.0 and from_w != 0.0):
            continue
        notional = abs(dw) * equity
        if notional < min_trade_notional:
            continue
        side = "buy" if dw > 0 else "sell"
        orders.append(
            {
                "symbol": sym,
                "side": side,
                "notional": round(notional, 2),
                "from_w": from_w,
                "to_w": to_w,
                "dry_run": True,
            }
        )

    # Deterministic ordering: sells before buys (free up buying power first), then symbol.
    orders.sort(key=lambda o: (o["side"] != "sell", o["symbol"]))
    return orders
