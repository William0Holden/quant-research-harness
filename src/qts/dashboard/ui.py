"""Shared look-and-feel: palette, CSS, plotly template, formatting helpers.

The theme base lives in .streamlit/config.toml; this module layers the pieces
Streamlit theming can't reach (metric cards, sidebar brand, chart template).
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG = "#0B0E14"
SURFACE = "#11161F"
BORDER = "#1E2633"
TEXT = "#E6EAF2"
MUTED = "#8B94A7"
ACCENT = "#5B8DEF"
GREEN = "#2ECC8F"
RED = "#F0616D"
AMBER = "#E8B44C"

_CSS = f"""
<style>
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}

.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1260px; }}

h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-size: 1.55rem !important; font-weight: 650 !important; }}
h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 1.0rem !important; font-weight: 600 !important; }}

/* KPI metric cards */
[data-testid="stMetric"] {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 0.85rem 1rem 0.75rem 1rem;
}}
[data-testid="stMetricLabel"] p {{
  color: {MUTED};
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 650; }}
[data-testid="stMetricDelta"] {{ font-size: 0.85rem; }}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: #0E1219;
  border-right: 1px solid {BORDER};
}}
.qts-brand {{
  display: flex; align-items: baseline; gap: 0.5rem;
  padding: 0.2rem 0 0.6rem 0;
  border-bottom: 1px solid {BORDER};
  margin-bottom: 0.4rem;
}}
.qts-brand .mark {{
  font-size: 1.25rem; font-weight: 750; letter-spacing: 0.02em; color: {TEXT};
}}
.qts-brand .sub {{ font-size: 0.72rem; color: {MUTED}; }}
.qts-side-note {{ color: {MUTED}; font-size: 0.74rem; line-height: 1.5; }}

/* Section eyebrow label */
.qts-eyebrow {{
  color: {MUTED}; font-size: 0.72rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.1em; margin: 0.2rem 0 0.1rem 0;
}}

/* Status dot */
.qts-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
</style>
"""

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12),
        colorway=[ACCENT, GREEN, AMBER, "#B48EF5", RED, "#4DC4D9"],
        xaxis=dict(gridcolor=BORDER, zeroline=False, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zeroline=False, linecolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=36, b=10),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER,
                        font=dict(color=TEXT)),
    )
)


def apply_theme() -> None:
    """Inject CSS and register the plotly template. Call once per run, in app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
    pio.templates["qts"] = _TEMPLATE
    pio.templates.default = "qts"


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="qts-eyebrow">{text}</div>', unsafe_allow_html=True)


def status_dot(ok: bool | None) -> str:
    color = GREEN if ok else (MUTED if ok is None else RED)
    return f'<span class="qts-dot" style="background:{color}"></span>'


def fmt_money(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "—"
    sign = "+" if (signed and x > 0) else ("−" if (signed and x < 0) else "")
    return f"{sign}${abs(x):,.2f}" if signed else f"${x:,.2f}"


def fmt_pct(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "—"
    sign = "+" if (signed and x > 0) else ""
    return f"{sign}{x * 100:.2f}%"
