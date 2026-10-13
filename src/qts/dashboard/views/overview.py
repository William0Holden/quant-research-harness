"""Overview — the money page: one chart that answers "how are we doing?", then the book.

The hero figure carries the equity curve, index-fund benchmarks, gross exposure and book
sentiment on one shared time axis, with every rebalance marked and its rationale in the
hover. Everything below it is detail you go looking for, not detail you need first.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import broker_data, hero, series_data, ui
from ..common import rebalance_records


def _latest_ok_record(recs: list[dict]) -> dict | None:
    for r in recs:
        if "error" not in r:
            return r
    return None


def _benchmark_return(bench: pd.DataFrame, symbol: str, dates) -> float | None:
    """Total return of one benchmark over exactly the account's own window.

    Shares `hero.rebase`'s date-grid alignment so the headline number and the chart can
    never tell different stories.
    """
    reb = hero.rebase(bench, symbol, dates, 1.0)
    if len(reb) < 2:
        return None
    return float(reb["value"].iloc[-1]) - 1.0


def render() -> None:
    st.title("Overview")
    st.caption("Alpaca **paper** account · weekly Friday rebalance, fills Monday open · "
               "research harness — not investment advice.")

    snap = broker_data.account_snapshot()
    hist = broker_data.equity_history()
    recs = rebalance_records()
    last_ok = _latest_ok_record(recs)
    pos = broker_data.positions()
    basis = broker_data.derived_basis()

    base = float(hist["equity"].iloc[0]) if len(hist) else None
    start_ts = hist["ts"].iloc[0] if len(hist) else None
    equity = snap["equity"] if snap else (float(hist["equity"].iloc[-1]) if len(hist) else None)
    earned = (equity - base) if (equity is not None and base) else None
    earned_pct = (earned / base) if (earned is not None and base) else None
    today = (snap["equity"] - snap["last_equity"]) if snap else None

    bench = (broker_data.benchmark_history(tuple(broker_data.BENCHMARKS), start_ts)
             if start_ts is not None else pd.DataFrame(columns=["ts", "symbol", "close"]))

    # ------------------------------------------------------------- headline
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Money earned (all-time)",
              ui.fmt_money(earned, signed=True) if earned is not None else "—",
              delta=ui.fmt_pct(earned_pct, signed=True) if earned_pct is not None else None)

    # Measured close-to-close on both sides. `equity` above is the LIVE intraday mark,
    # and comparing that to a benchmark that stops at last night's close would put a day
    # of the account's drift into the "excess return" for free.
    spy_ret = _benchmark_return(bench, "SPY", hist["ts"]) if len(hist) else None
    closed_ret = ((float(hist["equity"].iloc[-1]) / base - 1.0)
                  if (len(hist) and base) else None)
    if spy_ret is not None and closed_ret is not None:
        k2.metric("vs S&P 500", ui.fmt_pct(closed_ret - spy_ret, signed=True),
                  delta=f"index {ui.fmt_pct(spy_ret, signed=True)}", delta_color="off",
                  help="Excess return over SPY (total return, dividends reinvested) since "
                       "the account was funded. Both legs measured close-to-close over "
                       "the same sessions, so this trails the live equity by a day.")
    else:
        k2.metric("vs S&P 500", "—")

    k3.metric("Account equity", ui.fmt_money(equity),
              delta=(ui.fmt_money(today, signed=True) + " today") if today else None)

    invested = float(pos["market_value"].sum()) if len(pos) else 0.0
    inv_pct = (invested / equity) if (equity and invested) else None
    k4.metric("Invested", ui.fmt_pct(inv_pct) if inv_pct is not None else "—",
              delta=f"{len(pos)} positions · {ui.fmt_money(invested)}", delta_color="off",
              help="Gross exposure. The rest sits in cash — a book at 60% invested "
                   "beats a falling index by holding cash, not by picking stocks.")

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if last_ok is None:
            st.badge("no rebalance yet", color="grey")
        elif last_ok.get("kill_switch"):
            st.badge("KILL SWITCH LATCHED", icon=":material/warning:", color="red")
        else:
            st.badge("kill switch clear", icon=":material/verified:", color="green")
    with b2:
        if snap is None:
            st.badge("broker offline / no keys", color="orange")
        else:
            st.badge("broker connected", icon=":material/link:", color="blue")
    with b3:
        if last_ok:
            st.caption(f"Last rebalance **{last_ok['as_of']}** — "
                       f"{last_ok.get('n_orders', 0)} orders, "
                       f"{last_ok.get('n_order_errors', 0)} errors, "
                       f"equity source `{last_ok.get('equity_source', '?')}`")

    # ------------------------------------------------------------- hero
    exposure = series_data.exposure_series(recs)
    sentiment = (series_data.book_sentiment(recs, str(pd.Timestamp(start_ts).date()))
                 if start_ts is not None else pd.DataFrame())
    fig = hero.build_hero(hist, bench, exposure, sentiment, recs)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("▲ markers are rebalances — hover for the orders, turnover, regime "
                   "state and the signals that drove them. Green ok · amber degraded "
                   "regime read · red failed run or kill switch.")
    else:
        st.info("The chart builds up as the account trades — the first fills land at the "
                "next Monday open after a Friday rebalance.")

    # ------------------------------------------------------------- P&L attribution
    if basis.get("ok"):
        realized = basis["realized_pl"]
        unreal = float(pos["unrealized_pl"].sum()) if len(pos) else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Realized P&L", ui.fmt_money(realized, signed=True),
                  help="Booked on closed lots, recomputed from our own fill history.")
        c2.metric("Unrealized P&L", ui.fmt_money(unreal, signed=True),
                  help="Open positions, marked against fill-derived cost basis.")
        c3.metric("Realized + unrealized", ui.fmt_money(realized + unreal, signed=True),
                  help="Should reconcile to 'money earned' up to fees and intraday drift.")
        if basis.get("truncated"):
            st.warning("Fill history hit the paging limit — realized P&L and cost basis "
                       "are computed from a truncated history and are understated.")

    mismatched = pos[pos["basis_mismatch"]] if "basis_mismatch" in pos.columns else pos.iloc[:0]
    if len(mismatched):
        names = ", ".join(
            f"**{r.symbol}** (broker ${r.avg_entry_broker:,.2f} vs fills "
            f"${r.avg_entry:,.2f})" for r in mismatched.itertuples())
        st.warning(f"Broker cost basis disagrees with our fill history for {names}. "
                   "The figures below use the fill-derived basis; the broker's values "
                   "are kept in the table for comparison.", icon=":material/rule:")

    # ------------------------------------------------------------- book
    cL, cR = st.columns([3, 2])
    with cL:
        ui.eyebrow("Open positions")
        if len(pos):
            st.dataframe(
                pos, width="stretch", height=340, hide_index=True,
                column_order=["symbol", "qty", "avg_entry", "price", "market_value",
                              "unrealized_pl", "unrealized_plpc", "avg_entry_broker",
                              "basis_source"],
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "qty": st.column_config.NumberColumn("Qty", format="%.3f"),
                    "avg_entry": st.column_config.NumberColumn(
                        "Avg entry", format="$%.2f", help="Derived from our fill history."),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "unrealized_pl": st.column_config.NumberColumn("Unreal. P&L", format="$%.2f"),
                    "unrealized_plpc": st.column_config.NumberColumn("P&L %", format="percent"),
                    "avg_entry_broker": st.column_config.NumberColumn(
                        "Broker entry", format="$%.2f",
                        help="What the broker reports. Differs when its record is wrong."),
                    "basis_source": st.column_config.TextColumn("Basis"),
                })
            total_pl = float(pos["unrealized_pl"].sum())
            st.caption(f"Unrealized P&L across the book: **{ui.fmt_money(total_pl, signed=True)}**")
        else:
            st.caption("No open positions.")
    with cR:
        ui.eyebrow("Open orders")
        oo = broker_data.open_orders()
        if len(oo):
            st.dataframe(oo, width="stretch", height=340, hide_index=True,
                         column_config={"submitted": st.column_config.DatetimeColumn(
                             "Submitted (UTC)", format="MMM D, HH:mm")})
        else:
            st.caption("No open orders.")

    # ------------------------------------------------------------- rebalance history
    ui.eyebrow("Rebalance history")
    if recs:
        rows = []
        for r in recs:
            diag = r.get("diagnostics") or {}
            rs = diag.get("regime_state") or {}
            tp = r.get("trade_policy") or {}
            rows.append({
                "as_of": r.get("as_of"),
                "status": "failed" if "error" in r else
                          ("kill-switch" if r.get("kill_switch") else "ok"),
                "orders": r.get("n_orders"),
                "errors": r.get("n_order_errors"),
                "turnover": tp.get("turnover"),
                "gross": tp.get("gross", diag.get("gross")),
                "regime": ("unavailable" if (not rs or "error" in rs) else
                           "degraded" if rs.get("degraded", bool(rs.get("fetch_errors")))
                           else ("risk-off" if rs.get("risk_off") else "risk-on")),
                "equity": r.get("equity"),
                "note": r.get("error", ""),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={
                         "equity": st.column_config.NumberColumn("equity", format="$%.0f"),
                         "turnover": st.column_config.NumberColumn("turnover", format="percent"),
                         "gross": st.column_config.NumberColumn("gross", format="percent"),
                         "note": st.column_config.TextColumn("note", width="large"),
                     })
    else:
        st.caption("No rebalance runs recorded yet.")
