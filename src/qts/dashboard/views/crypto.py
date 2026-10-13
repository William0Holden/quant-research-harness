"""Crypto — the trend-timed BTC/ETH book, on its OWN paper account.

This page is deliberately honest about what the strategy is: RISK-MANAGED CRYPTO BETA, not
alpha. The headline it foregrounds is drawdown vs Bitcoin buy-and-hold, because that is the
only claim the evidence supports. The cross-sectional coin ranker was falsified before it
was built (beta ~1.0, no alpha); that story lives in the design doc, and this page does not
pretend otherwise.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import broker_data, ui
from ..common import crypto_rebalance_records


def _latest_ok(recs: list[dict]) -> dict | None:
    return next((r for r in recs if "error" not in r), None)


def render() -> None:
    st.title("Crypto")
    st.caption("Trend-timed **BTC/ETH** on a separate Alpaca **paper** account · "
               "daily eval, Tue–Thu execution · maker-first · "
               "**risk-managed beta, not alpha** — research harness, not investment advice.")

    snap = broker_data.crypto_account_snapshot()
    recs = crypto_rebalance_records()
    last = _latest_ok(recs)

    if snap is None and not recs:
        st.info("The crypto book has not run yet. It needs a **second** Alpaca paper "
                "account — set `APCA_CRYPTO_KEY_ID` / `APCA_CRYPTO_SECRET_KEY` from "
                "app.alpaca.markets → *Open New Paper Account*. Until then it refuses to "
                "trade (crypto and equities share one account, so each book would otherwise "
                "liquidate the other).")
        return

    # ------------------------------------------------------------- headline
    hist = broker_data.crypto_equity_history()
    base = float(hist["equity"].iloc[0]) if len(hist) else None
    equity = snap["equity"] if snap else (float(hist["equity"].iloc[-1]) if len(hist) else None)
    earned = (equity - base) if (equity is not None and base) else None
    today = (snap["equity"] - snap["last_equity"]) if snap else None

    k = st.columns(4)
    k[0].metric("P&L (all-time)",
                ui.fmt_money(earned, signed=True) if earned is not None else "—",
                delta=ui.fmt_pct(earned / base, signed=True) if (earned is not None and base) else None)
    k[1].metric("Account equity", ui.fmt_money(equity),
                delta=(ui.fmt_money(today, signed=True) + " today") if today else None)
    k[2].metric("Cash", ui.fmt_money(snap["cash"]) if snap else "—")
    pos = broker_data.crypto_positions()
    invested = float(pos["market_value"].sum()) if len(pos) else 0.0
    gross = (invested / equity) if (equity and invested) else 0.0
    k[3].metric("Gross exposure", f"{gross:.0%}",
                delta=ui.fmt_money(invested) + " deployed" if invested else "flat / all cash",
                delta_color="off")

    b = st.columns([1, 1, 2])
    with b[0]:
        if snap:
            st.badge(f"acct …{snap['account_number'][-4:]}", icon=":material/link:", color="blue")
        else:
            st.badge("broker offline", color="orange")
    with b[1]:
        if last and last.get("status") == "SKIPPED_WIDE_SPREAD":
            st.badge("rebalance skipped: wide spread", icon=":material/warning:", color="orange")
        elif last:
            st.badge("book live", icon=":material/verified:", color="green")
        else:
            st.badge("no rebalance yet", color="grey")
    with b[2]:
        if last:
            sz = last.get("sizing", {})
            st.caption(f"Last rebalance **{last['as_of']}** — gross "
                       f"{sz.get('gross', 0):.0%}, overlay "
                       f"{'on' if last.get('profile', {}).get('overlay_on') else 'off'} "
                       f"(m={sz.get('overlay_m', 1.0)})")

    # ------------------------------------------------- what it is (honest framing)
    with st.expander("What this book is — and what it is not", expanded=not recs):
        st.markdown(
            "**Claim: risk-managed crypto beta.** A 6-rule trend ensemble scales BTC/ETH "
            "exposure in and out; vol-targeted, gross capped at 1.0, maker-first execution. "
            "Backtested 2021–2026 net of real costs:\n\n"
            "| | CAGR | Sharpe | max drawdown | Calmar |\n"
            "|---|---|---|---|---|\n"
            "| **Trend book** | 12.0% | 0.58 | **−34%** | **0.35** |\n"
            "| BTC buy & hold | 15.9% | 0.54 | −77% | 0.21 |\n\n"
            "The Sharpe is a **dead heat** (the difference is inside its standard error). "
            "The product is the **drawdown — less than half of buy-and-hold**. "
            "A cross-sectional coin ranker was tested first and **falsified** (beta ≈ 1.0, "
            "no alpha vs the crypto market), so this book does not try to pick winners — it "
            "times exposure. It is not alpha, and it loses money when crypto does.")

    # ------------------------------------------------------------- equity curve
    ui.eyebrow("Account equity")
    if len(hist) >= 2:
        up = hist["equity"].iloc[-1] >= hist["equity"].iloc[0]
        fig = go.Figure(go.Scatter(
            x=hist["ts"], y=hist["equity"], mode="lines",
            line=dict(color=ui.GREEN if up else ui.RED, width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>"))
        fig.update_layout(height=280, showlegend=False,
                          yaxis=dict(tickprefix="$", tickformat=",.0f"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Equity history builds up as the book trades (daily, Tue–Thu window).")

    # ------------------------------------------------------------- positions + trend
    cL, cR = st.columns([3, 2])
    with cL:
        ui.eyebrow("Positions")
        if len(pos):
            st.dataframe(
                pos, width="stretch", hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "qty": st.column_config.NumberColumn("Qty", format="%.5f"),
                    "avg_entry": st.column_config.NumberColumn("Avg entry", format="$%.2f"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "unrealized_pl": st.column_config.NumberColumn("Unreal. P&L", format="$%.2f"),
                    "unrealized_plpc": st.column_config.NumberColumn("P&L %", format="percent"),
                })
        else:
            st.caption("Flat — the trend ensemble is defensive (mostly cash).")
    with cR:
        ui.eyebrow("Trend exposure (last rebalance)")
        if last:
            te = last.get("sizing", {}).get("trend_exposure", {})
            rv = last.get("sizing", {}).get("realized_vol", {})
            rows = [{"symbol": s,
                     "trend (of 6 rules)": None if te.get(s) is None else round(te[s], 3),
                     "realized vol": rv.get(s),
                     "weight": round(last.get("weights", {}).get(s, 0.0), 4)}
                    for s in last.get("target_weights", te)]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                             column_config={
                                 "realized vol": st.column_config.NumberColumn(format="percent"),
                                 "weight": st.column_config.NumberColumn(format="%.4f")})
            st.caption("Trend = fraction of 6 rules currently long. 0 → flat, 1 → fully "
                       "trend-on. Weight then scales by vol target (gross capped at 1.0).")

    # ---------------------------------------------- execution quality (cost ledger)
    if last and last.get("cost_ledger"):
        cl = last["cost_ledger"]
        if cl.get("n_fills"):
            ui.eyebrow("Execution quality — last rebalance")
            e = st.columns(4)
            e[0].metric("Fills", cl.get("n_fills"))
            mr = cl.get("maker_rate")
            e[1].metric("Maker rate", f"{mr:.0%}" if mr is not None else "—",
                        help="Share of notional filled as a resting (maker) order. Higher "
                             "= cheaper. Paper has no post-only flag, so this is measured "
                             "from the fee actually implied, never assumed.")
            fb = cl.get("fee_bps_effective")
            e[2].metric("Cost", f"{fb:.1f} bps" if fb is not None else "—",
                        help="Modelled round-trip cost per side. Alpaca **paper charges no "
                             "crypto fees**, so this is what the real fee schedule WOULD "
                             "charge — the honest number the broker's P&L hides.")
            adj = last.get("adjusted_equity")
            eq = last.get("equity")
            e[3].metric("Cost-adjusted NAV", ui.fmt_money(adj) if adj else "—",
                        delta=(ui.fmt_money(adj - eq, signed=True) + " vs broker")
                        if (adj and eq) else None, delta_color="off",
                        help="Broker equity minus the fees paper never charged. This, not "
                             "the account's reported equity, is the truth about performance.")

# ---------------------------------------------- sentiment signals (gated, accruing)
    sent = (last or {}).get("sentiment") if last else None
    if sent and "error" not in sent:
        ui.eyebrow("Sentiment signals — accruing, gated (no effect on trading)")
        st.caption("Three RISK signals (not direction): a vol-forecast lift, cross-asset "
                   "equity risk appetite, and an attention-spike detector. Each is logged "
                   "every run and earns un-gating only on its forward risk-IC record — "
                   "measured against volatility and drawdown, never returns. Values are "
                   "mostly blank until ~2–6 weeks of history accrue.")
        e = st.columns(3)
        era = sent.get("equity_risk_appetite")
        erc = sent.get("equity_risk_appetite_change")
        e[0].metric("Equity risk appetite",
                    f"{era:+.3f}" if isinstance(era, (int, float)) else "—",
                    help="Market-wide equity sentiment (cross-asset). A sharp DROP is the "
                         "risk signal — equity risk-off leads crypto drawdown.")
        e[1].metric("  ↳ recent change",
                    f"{erc:+.3f}" if isinstance(erc, (int, float)) else "—")
        pc = sent.get("per_coin", {})
        anyspike = any(v.get("spike") for v in pc.values())
        e[2].metric("Attention spike", "⚠️ yes" if anyspike else "no",
                    help="A surge in mention volume vs baseline — flags discrete events "
                         "(hacks, depegs, ETF decisions) faster than a daily rebalance.")
        rows = [{"coin": s,
                 "attention z": v.get("attention_z"),
                 "spike": v.get("spike"),
                 "vol forecast": v.get("vol_forecast"),
                 "trailing vol": v.get("trailing_vol")}
                for s, v in pc.items()]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                         column_config={
                             "vol forecast": st.column_config.NumberColumn(format="percent"),
                             "trailing vol": st.column_config.NumberColumn(format="percent")})
    # ------------------------------------------------------------- rebalance history
    ui.eyebrow("Rebalance history")
    if recs:
        rows = []
        for r in recs:
            sz = r.get("sizing", {})
            cl = r.get("cost_ledger", {})
            rows.append({
                "as_of": r.get("as_of"),
                "status": r.get("status", "?"),
                "gross": sz.get("gross"),
                "overlay_m": sz.get("overlay_m"),
                "orders": len(r.get("orders", [])),
                "fills": cl.get("n_fills"),
                "cost_bps": cl.get("fee_bps_effective"),
                "adj_equity": r.get("adjusted_equity"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={
                         "gross": st.column_config.NumberColumn(format="percent"),
                         "adj_equity": st.column_config.NumberColumn("adj_equity", format="$%.0f"),
                     })
    else:
        st.caption("No crypto rebalance runs recorded yet.")
