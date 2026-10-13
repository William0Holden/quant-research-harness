"""Target portfolio — what the live composite would trade at a chosen date."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from qts.features.ranker import rank_universe
from qts.risk.portfolio import build_target_portfolio

from .. import ui
from ..common import get_store, live_providers


def render() -> None:
    st.title("Target portfolio")
    st.caption("Sized long-only book from the live (gated) composite — the same provider "
               "set `qts.exec.rebalance` trades on the paper account.")
    store = get_store()

    c = st.columns([1.2, 1, 1, 1])
    pdate = c[0].date_input("As-of", value=date.today(), key="port_asof")
    equity = c[1].number_input("Equity ($)", 1000.0, 1e7, 100_000.0, step=1000.0, key="port_eq")
    regime_on = c[2].checkbox("Regime scaling", value=True, key="port_regime")
    go = c[3].button("Build portfolio", type="primary", key="port_go")

    if not go and "port_result" not in st.session_state:
        st.info("Pick a date and click **Build portfolio**.")
        return

    if go:
        with st.spinner("Ranking universe and sizing the book…"):
            syms = list(store.universe_as_of(pdate).symbol)
            ranked = rank_universe(store, live_providers(), syms, pdate, n_buckets=5)
            try:
                weights, diag = build_target_portfolio(
                    ranked, store, pdate, equity, regime_on=regime_on)
                st.session_state["port_result"] = (pdate, equity, weights, diag)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Portfolio build needs price history at this date: {e}")
                return

    pdate, equity, weights, diag = st.session_state["port_result"]

    m = st.columns(4)
    m[0].metric("Gross exposure", f"{diag.get('gross', 0):.0%}")
    m[1].metric("Positions", diag.get("n_positions", 0))
    ks = diag.get("kelly_scale")
    m[2].metric("Kelly scale", f"{ks:.2f}" if isinstance(ks, (int, float)) else "—")
    ro = (diag.get("regime_state") or {}).get("risk_off")
    m[3].metric("Regime", "risk-off" if ro else "risk-on")

    wdf = pd.DataFrame({"weight": weights}).sort_values("weight", ascending=False)
    wdf["$ alloc"] = (wdf["weight"] * equity).round(0)
    ui.eyebrow(f"Target book · {pdate}")
    st.dataframe(wdf.round(4), width="stretch", height=420,
                 column_config={"$ alloc": st.column_config.NumberColumn(format="$%.0f")})
    st.caption("Vol-targeted, quarter-Kelly sized off half the edge, per-position capped, "
               "regime-scaled. Cash = 1 − gross.")
