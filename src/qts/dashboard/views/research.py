"""Research — walk-forward backtest and per-provider signal IC."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qts.backtest.engine import run_backtest
from qts.backtest.metrics import equity_curve, summarize
from qts.features.base import backtestable_providers
from qts.pit.store import PITStore
from qts.research.ic import ic_summary, make_weights_fn

from .. import ui
from ..common import cached_backtest, cached_ic


def render() -> None:
    st.title("Research")
    tab_bt, tab_ic = st.tabs(["Walk-forward backtest", "Signal IC"])

    # ------------------------------------------------------------- backtest
    with tab_bt:
        cc = st.columns(5)
        start = cc[0].date_input("Start", value=date(2015, 1, 1), key="bt_start")
        end = cc[1].date_input("End", value=date(2024, 12, 31), key="bt_end")
        freq = cc[2].selectbox("Rebalance", ["ME", "W-FRI"],
                               format_func=lambda x: "Monthly" if x == "ME" else "Weekly",
                               key="bt_freq")
        nbk = cc[3].slider("Buckets", 3, 10, 5, key="bt_nb")
        run = cc[4].button("Run backtest", type="primary", key="bt_go")

        ui.eyebrow("Cost model — mandatory non-zero (leak guard #5)")
        cd = st.columns(3)
        spread = cd[0].number_input("Half-spread (bps/side)", 0.0, 50.0, 5.0, key="bt_spread")
        slippage = cd[1].number_input("Slippage (bps/side)", 0.0, 50.0, 3.0, key="bt_slip")
        borrow = cd[2].number_input("Short borrow (bps/yr)", 0.0, 1000.0, 50.0, key="bt_borrow")

        if run:
            periods, ppy, _ = cached_backtest(start, end, freq, nbk, spread, slippage, borrow)
            st.session_state["bt"] = (periods, ppy)

        if "bt" in st.session_state:
            periods, ppy = st.session_state["bt"]
            fig = go.Figure()
            for col, name in [("long_only_net", "Long-only (net)"),
                              ("long_short_net", "Long/short (net, research)"),
                              ("benchmark", "Equal-weight benchmark")]:
                eq = equity_curve(periods[col])
                fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name, mode="lines"))
            fig.update_layout(height=400, yaxis_title="Growth of $1")
            st.plotly_chart(fig, width="stretch")

            rows = []
            for col in ["long_only_net", "long_short_net", "benchmark"]:
                m = summarize(periods[col], ppy)
                rows.append({"strategy": col, **{k: m[k] for k in
                            ("cagr", "vol", "sharpe", "max_drawdown", "hit_rate", "n_periods")}})
            mt = pd.DataFrame(rows).set_index("strategy")
            for c in ["cagr", "vol", "max_drawdown", "hit_rate"]:
                mt[c] = (mt[c] * 100).round(1).astype(str) + "%"
            mt["sharpe"] = mt["sharpe"].round(2)
            st.dataframe(mt, width="stretch")
            st.caption(f"Avg turnover/rebalance {periods['turnover'].mean():.2f} · "
                       f"avg trade cost {periods['trade_cost'].mean()*1e4:.1f} bps · "
                       "long-only is the live surface; long/short is research-only.")
        else:
            st.info("Set parameters and click **Run backtest**.")

    # ------------------------------------------------------------- signal IC
    with tab_ic:
        c1, c2, c3, c4 = st.columns(4)
        ic_start = c1.date_input("Start", value=date(2022, 1, 1), key="ic_start")
        ic_end = c2.date_input("End", value=date(2025, 12, 31), key="ic_end")
        ic_freq = c3.selectbox("Rebalance freq", ["W-FRI", "ME"], key="ic_freq")
        ic_go = c4.button("Compute ICs", type="primary", key="ic_compute")

        if ic_go:
            icdf = cached_ic(ic_start, ic_end, ic_freq)
            st.session_state["icdf"] = (icdf, ic_start, ic_end, ic_freq)

        if "icdf" in st.session_state:
            icdf, ic_start, ic_end, ic_freq = st.session_state["icdf"]
            if icdf.empty:
                st.warning("No IC periods in range — widen the dates.")
                return
            st.dataframe(ic_summary(icdf).round(4).sort_values("t_stat", ascending=False),
                         width="stretch")
            st.caption("mean_ic > 0 with |t_stat| > 2 is a real (if small) edge. "
                       "Cumulative IC: a steadily rising line = consistent signal.")
            st.line_chart(icdf.fillna(0.0).cumsum())

            ui.eyebrow("Equal-weight vs walk-forward IC-weighted composite (net)")
            st.caption("IC weights at each rebalance use only ICs whose forward window fully "
                       "resolved before the rebalance date (trailing 26, negatives floored "
                       "at 0) — no look-ahead.")
            if st.button("Run comparison backtest", key="ic_bt"):
                s2 = PITStore(read_only=True)
                provs = backtestable_providers()
                eq = run_backtest(s2, provs, ic_start, ic_end, freq=ic_freq)
                icw = run_backtest(s2, provs, ic_start, ic_end, freq=ic_freq,
                                   weights_fn=make_weights_fn(icdf))
                s2.close()
                cmp_df = pd.DataFrame({
                    "equal_weight": (1 + eq.periods["long_only_net"]).cumprod(),
                    "ic_weighted": (1 + icw.periods["long_only_net"]).cumprod(),
                    "benchmark": (1 + eq.periods["benchmark"]).cumprod(),
                })
                st.line_chart(cmp_df)
                m = {"equal_weight": summarize(eq.periods["long_only_net"], eq.periods_per_year),
                     "ic_weighted": summarize(icw.periods["long_only_net"], icw.periods_per_year)}
                st.dataframe(pd.DataFrame(m).T.round(3), width="stretch")
        else:
            st.info("Set parameters and click **Compute ICs**.")
