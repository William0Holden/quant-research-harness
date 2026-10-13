"""Read-only Alpaca account data for the dashboard, cached and fault-tolerant.

Every fetcher returns plain dicts/DataFrames (picklable for st.cache_data) and
degrades to None/empty when keys are missing or the venue is unreachable — the
dashboard must render fine on a box with no broker credentials.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st


def has_keys() -> bool:
    return bool(os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"))


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(os.environ["APCA_API_KEY_ID"],
                         os.environ["APCA_API_SECRET_KEY"], paper=True)


@st.cache_data(ttl=60, show_spinner=False)
def account_snapshot() -> dict | None:
    """equity / last_equity / cash / buying_power, or None when unavailable."""
    if not has_keys():
        return None
    try:
        a = _client().get_account()
        return {
            "equity": float(a.equity),
            "last_equity": float(a.last_equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "currency": a.currency,
        }
    except Exception:  # noqa: BLE001  venue hiccup -> render without live numbers
        return None


def _session_dates(unix_seconds) -> pd.Series:
    """Alpaca stamps each daily equity point at the 20:00 ET close, which is 00:00 UTC
    the FOLLOWING day. Reading those as UTC dates shifted the whole curve forward one
    day -- Friday's close plotted on Saturday -- which silently misaligned every
    benchmark comparison and rebalance marker. Convert to Eastern before taking the date.
    """
    return (pd.to_datetime(pd.Series(unix_seconds), unit="s", utc=True)
              .dt.tz_convert("America/New_York")
              .dt.normalize()
              .dt.tz_localize(None))


@st.cache_data(ttl=300, show_spinner=False)
def equity_history() -> pd.DataFrame:
    """Daily account equity since inception (Alpaca portfolio history).

    Columns: ts (trading-session date), equity (float). Empty frame when unavailable.
    """
    if not has_keys():
        return pd.DataFrame(columns=["ts", "equity"])
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        h = _client().get_portfolio_history(
            GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
        df = pd.DataFrame({
            "ts": _session_dates(h.timestamp),
            "equity": [float(e) if e is not None else None for e in h.equity],
        }).dropna()
        return df[df["equity"] > 0].reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["ts", "equity"])


@st.cache_data(ttl=300, show_spinner=False)
def derived_basis() -> dict:
    """Cost basis recomputed from our own fill history.

    The broker's avg_entry_price is not trustworthy: on 2026-07-28 it reported HON at
    $49.95 against a $247.90 mark, a phantom +$509 that flipped the sign of the book's
    total unrealized P&L. Fills are the authoritative record of what we paid.

    Returns {"basis": {symbol: dict}, "realized_pl": float, "truncated": bool,
    "ok": bool} -- picklable for st.cache_data, and degrades to ok=False rather than
    breaking the page.
    """
    empty = {"basis": {}, "realized_pl": 0.0, "truncated": False, "ok": False}
    if not has_keys():
        return empty
    try:
        from qts.exec.cost_basis import account_basis
        rep = account_basis(paper=True)
        return {
            "basis": {s: p.as_dict() for s, p in rep.positions.items()},
            "realized_pl": rep.realized_pl,
            "truncated": rep.truncated,
            "ok": True,
        }
    except Exception:  # noqa: BLE001  fall back to broker-reported basis, flagged in the UI
        return empty


# A basis disagreement worth showing: more than a dollar AND more than 1% of the
# derived cost. Below that it is rounding, not corruption.
_BASIS_TOL_ABS = 1.0
_BASIS_TOL_REL = 0.01


@st.cache_data(ttl=60, show_spinner=False)
def positions() -> pd.DataFrame:
    """Open positions with unrealized P&L. Empty frame when flat/unavailable.

    `avg_entry`/`unrealized_pl` are the DERIVED (fill-based) figures when available;
    the broker's own numbers are kept alongside as `avg_entry_broker`/
    `unrealized_pl_broker` and flagged by `basis_mismatch` wherever they disagree, so a
    bad broker record is surfaced rather than silently overwritten.
    """
    cols = ["symbol", "qty", "avg_entry", "price", "market_value",
            "unrealized_pl", "unrealized_plpc", "avg_entry_broker",
            "unrealized_pl_broker", "basis_source", "basis_mismatch"]
    if not has_keys():
        return pd.DataFrame(columns=cols)
    try:
        basis = derived_basis().get("basis", {})
        rows = []
        for p in _client().get_all_positions():
            qty = float(p.qty)
            price = float(p.current_price) if p.current_price else None
            mv = float(p.market_value) if p.market_value else None
            b_entry = float(p.avg_entry_price)
            b_upl = float(p.unrealized_pl) if p.unrealized_pl else 0.0
            b_uplpc = float(p.unrealized_plpc) if p.unrealized_plpc else 0.0

            d = basis.get(p.symbol)
            # Only trust the derived basis when its quantity agrees with the broker's;
            # a qty mismatch means the fill history is incomplete for this symbol, and a
            # basis computed from partial fills is worse than the broker's.
            usable = bool(d) and abs(d["qty"] - qty) <= max(1e-6, abs(qty) * 1e-6)
            if usable and price is not None:
                entry = d["avg_cost"]
                upl = qty * price - d["cost_basis"]
                uplpc = (upl / d["cost_basis"]) if d["cost_basis"] else 0.0
                source = "fills"
                mismatch = (abs(entry - b_entry) > _BASIS_TOL_ABS
                            and abs(entry - b_entry) > _BASIS_TOL_REL * max(entry, 1e-9))
            else:
                entry, upl, uplpc = b_entry, b_upl, b_uplpc
                source = "broker"
                mismatch = False

            rows.append({
                "symbol": p.symbol, "qty": qty, "avg_entry": entry, "price": price,
                "market_value": mv, "unrealized_pl": upl, "unrealized_plpc": uplpc,
                "avg_entry_broker": b_entry, "unrealized_pl_broker": b_upl,
                "basis_source": source, "basis_mismatch": mismatch,
            })
        return (pd.DataFrame(rows, columns=cols)
                .sort_values("market_value", ascending=False, na_position="last")
                .reset_index(drop=True))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)


@st.cache_data(ttl=60, show_spinner=False)
def open_orders() -> pd.DataFrame:
    cols = ["submitted", "symbol", "side", "amount", "status"]
    if not has_keys():
        return pd.DataFrame(columns=cols)
    try:
        from alpaca.trading.requests import GetOrdersRequest
        orders = _client().get_orders(GetOrdersRequest(status="open", limit=500))
        rows = [{
            "submitted": pd.Timestamp(o.submitted_at).tz_convert("UTC").tz_localize(None),
            "symbol": o.symbol,
            "side": o.side.value,
            "amount": (f"${float(o.notional):,.2f}" if o.notional
                       else f"{o.qty} sh"),
            "status": o.status.value,
        } for o in orders]
        return pd.DataFrame(rows, columns=cols).sort_values(
            "submitted", ascending=False).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------
# Benchmarks. "Are we beating an index fund?" is the first question anyone asks of an
# equity curve, and it cannot be answered from the curve alone. Alpaca's market-data
# API is already authorised by the trading keys, so this costs no new credential.
# --------------------------------------------------------------------------
BENCHMARKS = {
    "SPY": "S&P 500",
    "RSP": "S&P 500 equal-weight",
    "SGOV": "T-bills (cash)",
}


@st.cache_data(ttl=900, show_spinner=False)
def benchmark_history(symbols: tuple[str, ...], start: pd.Timestamp) -> pd.DataFrame:
    """Daily dividend-adjusted closes for benchmark ETFs.

    Long format: ts, symbol, close. Empty frame without keys or on any venue failure —
    the hero chart drops the benchmark traces and still renders the equity curve.
    """
    cols = ["ts", "symbol", "close"]
    if not has_keys() or not symbols:
        return pd.DataFrame(columns=cols)
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"],
                                           os.environ["APCA_API_SECRET_KEY"])
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start).to_pydatetime(),
            adjustment="all",          # total return: splits AND dividends
        ))
        df = bars.df.reset_index()
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df.rename(columns={"timestamp": "ts"})
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
        return df[["ts", "symbol", "close"]].sort_values(["symbol", "ts"]).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------
# Crypto book — a SEPARATE Alpaca paper account (APCA_CRYPTO_* keys). It must never
# read the equity keys: the two books live in different accounts precisely so neither
# can see or trade the other's positions. Every fetcher degrades to None/empty when the
# crypto account is not configured, so the page renders on a box with equity keys only.
# --------------------------------------------------------------------------
def has_crypto_keys() -> bool:
    return bool(os.environ.get("APCA_CRYPTO_KEY_ID")
                and os.environ.get("APCA_CRYPTO_SECRET_KEY"))


def _crypto_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(os.environ["APCA_CRYPTO_KEY_ID"],
                         os.environ["APCA_CRYPTO_SECRET_KEY"], paper=True)


@st.cache_data(ttl=60, show_spinner=False)
def crypto_account_snapshot() -> dict | None:
    if not has_crypto_keys():
        return None
    try:
        a = _crypto_client().get_account()
        return {"equity": float(a.equity), "last_equity": float(a.last_equity),
                "cash": float(a.cash), "account_number": a.account_number}
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=300, show_spinner=False)
def crypto_equity_history() -> pd.DataFrame:
    if not has_crypto_keys():
        return pd.DataFrame(columns=["ts", "equity"])
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        h = _crypto_client().get_portfolio_history(
            GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
        df = pd.DataFrame({
            "ts": _session_dates(h.timestamp),
            "equity": [float(e) if e is not None else None for e in h.equity],
        }).dropna()
        return df[df["equity"] > 0].reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["ts", "equity"])


@st.cache_data(ttl=60, show_spinner=False)
def crypto_positions() -> pd.DataFrame:
    cols = ["symbol", "qty", "avg_entry", "price", "market_value",
            "unrealized_pl", "unrealized_plpc"]
    if not has_crypto_keys():
        return pd.DataFrame(columns=cols)
    try:
        rows = [{
            "symbol": p.symbol, "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "price": float(p.current_price) if p.current_price else None,
            "market_value": float(p.market_value) if p.market_value else None,
            "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else 0.0,
            "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else 0.0,
        } for p in _crypto_client().get_all_positions()]
        return (pd.DataFrame(rows, columns=cols)
                .sort_values("market_value", ascending=False, na_position="last")
                .reset_index(drop=True))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)
