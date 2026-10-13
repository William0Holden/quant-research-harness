"""QTS dashboard — app shell: auth gate, theme, navigation.

Run:  uv run streamlit run src/qts/dashboard/app.py --server.address 0.0.0.0 --server.port 8501
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

from qts.dashboard import ui
from qts.dashboard.common import corpus_status, rebalance_records
from qts.dashboard.views import (crypto, data, overview, portfolio, research, signals,
                                 system)

st.set_page_config(page_title="QTS", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")


def _check_password() -> None:
    """Optional gate. If QTS_DASH_PASSWORD is set (e.g. when publicly exposed), require it.
    Unset (LAN/dev) => open. Constant-time compare; never stores the password in state."""
    pw = os.environ.get("QTS_DASH_PASSWORD")
    if not pw or st.session_state.get("_authed"):
        return
    st.title("QTS")
    entered = st.text_input("Password", type="password", key="_pw")
    if not entered:
        st.stop()
    if hmac.compare_digest(entered, pw):
        st.session_state["_authed"] = True
        st.rerun()
    st.error("Incorrect password")
    st.stop()


_check_password()
ui.apply_theme()

pages = [
    st.Page(overview.render, title="Overview", icon=":material/dashboard:",
            url_path="overview", default=True),
    st.Page(portfolio.render, title="Target portfolio", icon=":material/pie_chart:",
            url_path="portfolio"),
    st.Page(crypto.render, title="Crypto", icon=":material/currency_bitcoin:",
            url_path="crypto"),
    st.Page(signals.render, title="Signals", icon=":material/insights:",
            url_path="signals"),
    st.Page(research.render, title="Research", icon=":material/science:",
            url_path="research"),
    st.Page(data.render, title="Data", icon=":material/database:", url_path="data"),
    st.Page(system.render, title="System", icon=":material/monitor_heart:",
            url_path="system"),
]

with st.sidebar:
    st.markdown('<div class="qts-brand"><span class="mark">QTS</span>'
                '<span class="sub">NLP + factor harness</span></div>',
                unsafe_allow_html=True)

nav = st.navigation(pages)

with st.sidebar:
    recs = rebalance_records()
    last = next((r for r in recs if "error" not in r), None)
    cs = corpus_status()
    bits = []
    if last:
        bits.append(f"Last rebalance <b>{last['as_of']}</b> · {last.get('n_orders', 0)} orders")
    if cs:
        bits.append(f"Corpus {cs.get('news_items', 0):,} news / "
                    f"{cs.get('social_posts', 0):,} posts")
    bits.append("Paper account · not investment advice")
    st.markdown('<div class="qts-side-note">' + "<br>".join(bits) + "</div>",
                unsafe_allow_html=True)

nav.run()
