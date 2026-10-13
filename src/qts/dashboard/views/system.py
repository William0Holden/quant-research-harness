"""System — leak guards, macro regime, and ops health (cron/collector freshness)."""
from __future__ import annotations

import time
from datetime import date

import streamlit as st

from qts.checks import run_leak_checks
from qts.config import OUTPUTS_DIR
from qts.risk.regime import regime_state

from .. import ui
from ..common import get_store

# (label, path relative to repo, staleness threshold in hours)
_WATCH = [
    ("Daily pipeline", "pipeline.log", 30),
    ("Rebalance runner", "rebalance.log", 8 * 24),
    ("News collector", "news.log", 3),
    ("Reddit collector", "reddit.log", 3),
]


def render() -> None:
    st.title("System")
    store = get_store()

    # ------------------------------------------------------------- ops health
    ui.eyebrow("Ops health — cron & collector freshness")
    rows = []
    now = time.time()
    for label, rel, max_h in _WATCH:
        p = OUTPUTS_DIR / rel
        if p.exists():
            age_h = (now - p.stat().st_mtime) / 3600
            ok = age_h <= max_h
            rows.append((label, ok, f"{age_h:.1f}h ago (limit {max_h}h)"))
        else:
            rows.append((label, None, "no log yet"))
    for label, ok, detail in rows:
        st.markdown(f"{ui.status_dot(ok)} **{label}** — {detail}",
                    unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------- leak guards
    ui.eyebrow("Leak-guard self-checks (run live against the store)")
    checks = run_leak_checks(store)
    n_pass = int(checks["passed"].sum())
    (st.success if n_pass == len(checks) else st.error)(
        f"{n_pass}/{len(checks)} leak guards passing")
    for _, row in checks.iterrows():
        st.markdown(f"{ui.status_dot(bool(row['passed']))} **{row['check']}** — "
                    f"{row['detail']}", unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------- regime
    ui.eyebrow("Macro regime conditioner (FRED, key-free)")
    try:
        rs = regime_state(date.today())
        g = st.columns(4)
        g[0].metric("HY OAS (credit)", rs.get("hy_oas", "—"))
        g[1].metric("Term spread 10y−2y", rs.get("term_spread", "—"))
        g[2].metric("VIX", rs.get("vix", "—"))
        g[3].metric("Gross multiplier", rs.get("gross_multiplier", "—"))
        (st.error if rs.get("risk_off") else st.success)(
            "RISK-OFF — gross scaled down" if rs.get("risk_off")
            else "RISK-ON — full gross")
        with st.expander("Raw regime state"):
            st.json(rs)
        st.caption("Scales gross exposure in the sizing layer (momentum crashes in "
                   "credit-stress / high-vol regimes). Not a ranker input.")
    except Exception as e:  # noqa: BLE001
        st.warning(f"FRED fetch failed (network): {e}")
