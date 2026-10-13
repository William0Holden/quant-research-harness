"""Broker interface (wired seam; paper-first).

Alpaca is the prototype/paper/first-live surface (commission-free + fractional makes
small-capital weekly rebalancing economic). IBKR is a later port behind the same Broker
ABC. Live trading is gated behind a proven paper stage — this v1 ships the interface and a
DryRunBroker only; no order ever reaches a venue without explicit wiring + keys.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qts.exec.reconcile import compute_orders


def to_venue(symbol: str) -> str:
    """Universe (Yahoo) class-share form -> Alpaca form: BRK-B -> BRK.B, BF-B -> BF.B.

    Every weekly rebalance from 07-31 to 08-28 sent "BRK-B" and got back
    `asset "BRK-B" not found` -- the name simply never traded."""
    return symbol.replace("-", ".")


def from_venue(symbol: str) -> str:
    """Inverse of to_venue, so positions key on the same symbols the universe uses."""
    return symbol.replace(".", "-")


@dataclass
class TargetPortfolio:
    """Output of the sizing/risk layer: symbol -> target weight (long-only in v1)."""
    weights: dict[str, float] = field(default_factory=dict)


class Broker(ABC):
    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...

    @abstractmethod
    def rebalance(
        self, target: TargetPortfolio, dry_run: bool = True, band_w: float = 0.0
    ) -> list[dict]:
        """Return the orders that WOULD be / were placed. dry_run must default True."""
        ...

    def filter_tradable(self, symbols: list[str]) -> tuple[list[str], dict[str, str]]:
        """(tradable symbols, {untradable symbol: reason}). Default: everything trades.

        Called BEFORE ranking so a name the venue cannot trade never receives weight.
        Otherwise the sizer hands it a slice of the book and that slice sits in cash
        behind a weekly order error (EA at 2.9% of target, Aug 2026)."""
        return list(symbols), {}


class DryRunBroker(Broker):
    """Logs intended orders; never contacts a venue. The only broker available in v1.

    Delegates to the same `compute_orders` core the Alpaca path uses, against a
    nominal $100,000 research equity (there is no real account to size against),
    so the audit trail has ONE order-dict shape regardless of broker:
    {symbol, side, notional, from_w, to_w, dry_run}.
    """

    NOMINAL_EQUITY = 100_000.0

    def __init__(self):
        self._positions: dict[str, float] = {}

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def rebalance(
        self, target: TargetPortfolio, dry_run: bool = True, band_w: float = 0.0
    ) -> list[dict]:
        return compute_orders(self._positions, target.weights,
                              equity=self.NOMINAL_EQUITY, min_trade_notional=1.0,
                              band_w=band_w)

class AlpacaBroker(Broker):
    """Alpaca paper/live broker (paper by default).

    To go LIVE on paper trading the user must, AFTER this code is in place:
        1. `uv pip install alpaca-py`
        2. export APCA_API_KEY_ID=<paper key id>
           export APCA_API_SECRET_KEY=<paper secret>
           (get both from https://app.alpaca.markets/ -> Paper Trading -> API Keys)

    Until then, the module imports fine and ALL pure logic (rebalance dry-run,
    order computation) works without the keys or the package. `alpaca` is
    lazy-imported inside methods, and keys are only required when a method that
    actually contacts the venue is called. Endpoint defaults to paper (paper=True).
    """

    def __init__(
        self,
        key_id: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        min_trade_notional: float = 1.0,
    ):
        # Keys are read lazily/optionally — do NOT raise here so construction and
        # dry-run rebalancing work with no credentials present.
        self._key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        self._secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        self.paper = paper
        self.min_trade_notional = min_trade_notional
        self._client = None  # cached TradingClient, built on first live call
        self._data_client = None  # cached StockHistoricalDataClient (whole-share fallback)
        self._tradable_cache: set[str] | None = None

    # --- live plumbing (only invoked when actually talking to the venue) ---
    def _trading_client(self):
        """Build (and cache) an Alpaca TradingClient. Raises if package/keys missing."""
        if self._client is not None:
            return self._client
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:  # package not installed
            raise RuntimeError(
                "alpaca-py is not installed. Run `uv pip install alpaca-py` to enable "
                "live/paper trading against Alpaca."
            ) from e
        if not self._key_id or not self._secret_key:
            raise RuntimeError(
                "Alpaca API keys are not set. Export APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY (Paper Trading keys from app.alpaca.markets) "
                "before calling a live broker method."
            )
        self._client = TradingClient(
            self._key_id, self._secret_key, paper=self.paper
        )
        return self._client

    def get_positions(self) -> dict[str, float]:
        """Return symbol -> current weight (position market value / account equity).

        EQUITIES ONLY. Alpaca holds crypto and equities in the SAME account, and a symbol
        absent from this book's target weights has an IMPLIED TARGET OF ZERO -- so if a
        crypto position ever appeared here, the next equity rebalance would dutifully
        compute a sell and liquidate it. Filtering by asset_class is the belt to the
        crypto profile's braces (which keeps the two books in separate accounts entirely).
        Skipped positions are reported, never silently dropped.
        """
        client = self._trading_client()
        account = client.get_account()
        equity = float(account.equity)
        if equity <= 0:
            return {}
        positions = client.get_all_positions()
        weights: dict[str, float] = {}
        skipped: list[str] = []
        for p in positions:
            if str(getattr(p, "asset_class", "us_equity")).endswith("crypto"):
                skipped.append(p.symbol)
                continue
            weights[from_venue(p.symbol)] = float(p.market_value) / equity
        if skipped:
            print(f"  ! equity book ignoring {len(skipped)} crypto position(s): "
                  f"{', '.join(skipped)} (they belong to the crypto book, not this one)")
        return weights

    def _account_equity(self) -> float:
        client = self._trading_client()
        return float(client.get_account().equity)

    def _tradable_symbols(self) -> set[str]:
        """Venue-form symbols of every ACTIVE, tradable US equity. One call, cached."""
        if self._tradable_cache is None:
            from alpaca.trading.enums import AssetClass, AssetStatus
            from alpaca.trading.requests import GetAssetsRequest
            assets = self._trading_client().get_all_assets(
                GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
            self._tradable_cache = {a.symbol for a in assets if getattr(a, "tradable", True)}
        return self._tradable_cache

    def filter_tradable(self, symbols: list[str]) -> tuple[list[str], dict[str, str]]:
        """Reconcile the universe against the venue's asset list.

        A universe name that Alpaca has no active, tradable asset for is dropped with a
        reason -- delisted/acquired names (EA, AVB, EQR in Aug 2026) and any symbol the
        venue spells differently. Raises on venue failure; the caller decides whether
        to fail open."""
        live = self._tradable_symbols()
        ok, bad = [], {}
        for sym in symbols:
            if to_venue(sym) in live:
                ok.append(sym)
            else:
                bad[sym] = "no active tradable asset on venue"
        return ok, bad

    def _latest_trade_price(self, symbol: str) -> float:
        """Latest trade price from the (free, IEX-fed) data API — same keys as trading."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(self._key_id, self._secret_key)
        trades = self._data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(trades[symbol].price)

    def rebalance(
        self, target: TargetPortfolio, dry_run: bool = True, band_w: float = 0.0
    ) -> list[dict]:
        """Move from current positions to `target` weights via fractional/notional orders.

        dry_run=True (default) computes and returns the order list WITHOUT submitting.
        dry_run=False submits each order through the Alpaca TradingClient.
        """
        current = self.get_positions()
        equity = self._account_equity()
        orders = compute_orders(
            current,
            target.weights,
            equity,
            min_trade_notional=self.min_trade_notional,
            band_w=band_w,
        )

        if dry_run:
            return orders

        # Live submission path. Each order is isolated: one venue rejection must
        # never strand a half-submitted rebalance (2026-07-03: a single
        # "not fractionable" rejection on HON aborted the loop with ~90 names
        # already on the book and the rest never sent).
        from alpaca.common.exceptions import APIError
        from alpaca.trading.client import TradingClient  # noqa: F401  (ensures import error surfaces consistently)
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        client = self._trading_client()
        submitted: list[dict] = []
        for o in orders:
            side = OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL
            try:
                # Sell-to-zero (full liquidation) is cleaner via close_position, which
                # avoids fractional rounding leaving dust.
                venue_sym = to_venue(o["symbol"])
                if o["side"] == "sell" and abs(o["to_w"]) < 1e-12:
                    client.close_position(venue_sym)
                    placed = dict(o, dry_run=False, qty="all", status="submitted")
                else:
                    try:
                        client.submit_order(MarketOrderRequest(
                            symbol=venue_sym,
                            notional=round(o["notional"], 2),
                            side=side,
                            time_in_force=TimeInForce.DAY,
                        ))
                        placed = dict(o, dry_run=False, status="submitted")
                    except APIError as e:
                        if "not fractionable" not in str(e):
                            raise
                        # Notional (fractional) orders are rejected on assets Alpaca
                        # marks non-fractionable — resubmit as whole shares, rounded
                        # down at the latest trade price.
                        price = self._latest_trade_price(venue_sym)
                        qty = int(o["notional"] // price)
                        if qty < 1:
                            placed = dict(o, dry_run=False, status="skipped",
                                          reason="non-fractionable and notional "
                                                 f"below one share (~${price:.2f})")
                        else:
                            client.submit_order(MarketOrderRequest(
                                symbol=venue_sym, qty=qty, side=side,
                                time_in_force=TimeInForce.DAY,
                            ))
                            placed = dict(o, dry_run=False, status="submitted", qty=qty)
            except APIError as e:
                placed = dict(o, dry_run=False, status="error", error=str(e))
            submitted.append(placed)
        return submitted
