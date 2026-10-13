"""Signals — cross-sectional rankings, NLP features, and gated alt-data signals."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from qts.features.ranker import rank_universe

from .. import ui
from ..common import get_providers, get_store, rebalance_records


def render() -> None:
    st.title("Signals")
    store = get_store()

    tab_rank, tab_nlp, tab_gated = st.tabs(["Rankings", "NLP features", "Gated alt-data"])

    # ------------------------------------------------------------- rankings
    with tab_rank:
        c = st.columns([1.2, 1, 1])
        rank_date = c[0].date_input("Rank as-of", value=date.today(), key="rank_asof")
        nb = c[1].slider("Buckets", 3, 10, 5, key="nb_rank")
        go = c[2].button("Rank universe", type="primary", key="rank_go")
        st.caption("Research view: includes IC-gated and forward-only signals not eligible "
                   "for the live composite (the Target portfolio page shows what trades).")
        if go:
            with st.spinner("Ranking…"):
                syms = list(store.universe_as_of(rank_date).symbol)
                r = rank_universe(store, get_providers(), syms, rank_date,
                                  n_buckets=nb).dropna(subset=["score"])
                st.session_state["rank_result"] = (rank_date, nb, r)
        if "rank_result" in st.session_state:
            rank_date, nb, r = st.session_state["rank_result"]
            longs = r.index[r["bucket"] == nb].tolist()
            shorts = r.index[r["bucket"] == 1].tolist()
            cL, cS = st.columns(2)
            cL.success(f"**Top bucket ({len(longs)})** — {', '.join(longs[:40])}"
                       + (" …" if len(longs) > 40 else ""))
            cS.error(f"**Bottom bucket ({len(shorts)})** — {', '.join(shorts[:40])}"
                     + (" …" if len(shorts) > 40 else ""))
            show_cols = ["score", "bucket"] + [c for c in r.columns if c.startswith("z_")]
            st.dataframe(r[show_cols].round(3), width="stretch", height=430)
            st.caption("z_* are sign-adjusted standardized features (higher = more bullish); "
                       "composite = equal-weight mean over available features.")
        else:
            st.info("Pick a date and click **Rank universe**.")

    # ------------------------------------------------------------- NLP
    with tab_nlp:
        cnt = store.con.execute(
            "SELECT concept, COUNT(*) n FROM nlp_features GROUP BY concept").df()
        if cnt.empty:
            st.info("No NLP features yet — the Sunday GPU worker batch populates them "
                    "(`uv run python -m qts.nlp.factory`).")
        else:
            ui.eyebrow("Feature coverage")
            st.write({r["concept"]: int(r["n"]) for _, r in cnt.iterrows()})
            nd = st.date_input("As-of", value=date.today(), key="nlp_asof")
            syms = list(store.universe_as_of(nd).symbol)
            sent = store.nlp_features_as_of(syms, "filing_sentiment", nd)
            diff = store.nlp_features_as_of(syms, "lazy_prices_diff", nd)
            ndf = (pd.DataFrame({"filing_sentiment": sent, "lazy_prices_diff": diff})
                   .dropna(how="all").sort_values("filing_sentiment", ascending=False))
            st.dataframe(ndf.round(3), width="stretch", height=420)
            st.caption("filing_sentiment: LLM tone of the latest filing (higher = bullish). "
                       "lazy_prices_diff: YoY language-shift magnitude (higher = bearish). "
                       "Both PIT-gated on filing date.")

    # ------------------------------------------------------------- gated alt-data
    with tab_gated:
        st.caption("New providers compute into every rebalance audit at **weight zero** "
                   "until live IC proves them out — this is the paper trail.")
        recs = [r for r in rebalance_records() if r.get("gated_signals")]
        if not recs:
            st.info("No gated signals recorded yet — they appear after the next rebalance.")
        else:
            latest = recs[0]
            ui.eyebrow(f"Gated signals · rebalance {latest['as_of']}")
            gs = latest["gated_signals"]
            frames = {}
            for name, vals in gs.items():
                if isinstance(vals, dict) and "_error" not in vals:
                    frames[name] = pd.Series(vals, dtype=float)
                else:
                    st.warning(f"`{name}` errored: {vals.get('_error', 'unknown')}")
            if frames:
                gdf = pd.DataFrame(frames)
                st.dataframe(gdf.round(4).sort_index(), width="stretch", height=420)
                st.caption(f"{len(gdf)} symbols with at least one gated signal. Anomaly "
                           "providers stay NaN until ~21 days of collector history exists.")
            else:
                st.info("Gated providers ran but produced no non-NaN values yet "
                        "(collectors are still accumulating history).")
