"""Data — universe, price coverage, and the alt-data corpus."""
from __future__ import annotations

from datetime import date

import streamlit as st

from .. import ui
from ..common import corpus_status, get_store


def render() -> None:
    st.title("Data")
    store = get_store()
    counts = store.table_counts()

    k = st.columns(4)
    k[0].metric("Universe rows", f"{counts['universe']:,}")
    k[1].metric("Price bars", f"{counts['prices']:,}")
    k[2].metric("Fundamental facts", f"{counts['fundamentals']:,}")
    k[3].metric("Filing events", f"{counts['events']:,}")

    ui.eyebrow("Survivorship-free universe")
    as_of = st.date_input("Universe as-of", value=date(2022, 6, 1),
                          min_value=date(2005, 1, 1), max_value=date.today(),
                          key="uni_asof")
    u = store.universe_as_of(as_of)
    st.write(f"**{len(u)} names alive on {as_of}** (includes names that later delisted).")
    if "FRC" in set(u["symbol"]):
        st.success("First Republic (FRC) is present — it had not yet failed. Move the date "
                   "past 2023-05 and it correctly disappears (no survivorship leak).")
    st.dataframe(u, width="stretch", height=260)

    ui.eyebrow("Price coverage")
    st.dataframe(store.price_coverage(), width="stretch", height=260)

    cs = corpus_status()
    if cs:
        ui.eyebrow("Alt-data corpus (news + social collectors)")
        c = st.columns(4)
        c[0].metric("News items", f"{cs.get('news_items', 0):,}")
        c[1].metric("Social posts", f"{cs.get('social_posts', 0):,}")
        c[2].metric("Last news run", str(cs.get("last_news_run", "never"))[:16])
        c[3].metric("Last reddit run", str(cs.get("last_reddit_run", "never"))[:16])
        st.caption("Collectors run on cron (news hourly, reddit twice hourly) into a "
                   "separate corpus DB; a nightly aggregation distills PIT `alt_signals`.")
