"""The one chart that answers "how are we doing?" without a follow-up question.

Three x-linked rows:

  1. Growth of the account, against index funds rebased to the same starting dollar,
     with a marker at every rebalance carrying its full rationale in the hover.
  2. Gross exposure — the row that explains the divergence in row 1. A book sitting at
     58% invested beats a falling index by holding cash, not by picking stocks, and
     without this row that distinction is invisible.
  3. Sentiment of the names we actually hold, against the universe as a reference.

Pure functions of DataFrames in, `go.Figure` out — no Streamlit, no network, so the
layout is testable without a browser. Each feed degrades independently: rows with no
data are dropped and the remaining rows re-proportion.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import ui

BENCH_STYLE = {
    "SPY": dict(color="#8B94A7", dash="solid", label="S&P 500 (SPY)", rank=3),
    "RSP": dict(color="#B48EF5", dash="dot", label="S&P 500 equal-weight (RSP)", rank=4),
    "SGOV": dict(color="#4DC4D9", dash="dash", label="T-bills (SGOV)", rank=5),
}

_MARKER_OK = ui.GREEN
_MARKER_WARN = ui.AMBER
_MARKER_FAIL = ui.RED

# Only these two feed a tripwire (see risk/regime.DECISION_SERIES); term_spread failing
# is noise, not a blind spot, and naming it in a "we can't see the regime" message would
# be misleading.
_DECISION_SERIES = ("hy_oas", "vix")

# Provider ids are snake_case with embedded numbers -- a blind underscore strip turns
# momentum_12_1 into "momentum 12 1".
_SIGNAL_LABELS = {
    "momentum_12_1": "12-1 momentum",
    "low_volatility": "low volatility",
    "earnings_yield": "earnings yield",
    "gross_profitability": "gross profitability",
    "net_issuance": "net issuance",
    "insider_activity": "insider activity",
    "event8k_intensity": "8-K intensity",
    "restatement_flag": "restatements",
    "short_interest": "short interest",
    "filing_sentiment": "filing sentiment",
    "lazy_prices_diff": "filing-text change",
}


# --------------------------------------------------------------------------- #
# rationale
# --------------------------------------------------------------------------- #
def _fmt_pct(x, digits=0) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _regime_line(rec: dict) -> str:
    diag = rec.get("diagnostics") or {}
    rs = diag.get("regime_state") or {}
    if not rs or "error" in rs:
        return "Regime: <b>UNAVAILABLE</b> — brake could not be evaluated"
    mult = diag.get("regime_gross_multiplier", rs.get("gross_multiplier", 1.0))
    degraded = rs.get("degraded", bool(rs.get("fetch_errors")))
    if degraded:
        blind = [k for k in _DECISION_SERIES
                 if k in set(rs.get("fetch_errors") or []) | set(rs.get("stale_series") or [])]
        missing = ", ".join(blind or rs.get("fetch_errors") or ["?"])
        # Runs from before the 2026-07-28 fail-closed change sized at full gross with an
        # unreadable brake. Say so plainly rather than reporting "capped ×1".
        tail = (f"gross capped ×{mult:g}" if mult < 1.0
                else "<b>brake did not engage</b> (run predates the fail-closed fix)")
        return f"Regime: <b>DEGRADED</b> ({missing} unreadable) — {tail}"
    state = "risk-OFF" if rs.get("risk_off") else "risk-on"
    bits = []
    if pd.notna(rs.get("vix")):
        bits.append(f"VIX {rs['vix']:.1f}")
    if pd.notna(rs.get("hy_oas")):
        bits.append(f"HY OAS {rs['hy_oas']:.2f}")
    detail = f" ({', '.join(bits)})" if bits else ""
    return f"Regime: <b>{state}</b>{detail} — gross ×{mult:g}"


def _top_signals(rec: dict, n: int = 3) -> str:
    pw = rec.get("provider_weights") or {}
    if not pw:
        return ""
    top = sorted(pw.items(), key=lambda kv: -float(kv[1]))[:n]
    pretty = " · ".join(
        f"{_SIGNAL_LABELS.get(k, k.replace('_', ' '))} {_fmt_pct(v)}" for k, v in top)
    return f"Top signals: {pretty}"


def _biggest_trades(rec: dict, n: int = 3) -> list[str]:
    """The trades that actually moved the book, by weight change — not by dollar size,
    since a big notional in a big position can be a trivial reweighting."""
    orders = rec.get("orders") or []
    moves = []
    for o in orders:
        try:
            d = float(o["to_w"]) - float(o["from_w"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(d) > 1e-6:
            moves.append((o.get("symbol", "?"), d))
    if not moves:
        return []
    moves.sort(key=lambda t: -t[1])
    buys = [m for m in moves if m[1] > 0][:n]
    sells = [m for m in moves if m[1] < 0][-n:][::-1]
    out = []
    if buys:
        out.append("Biggest buys: " + " · ".join(
            f"{s} +{d * 100:.2f}%" for s, d in buys))
    if sells:
        out.append("Biggest sells: " + " · ".join(
            f"{s} {d * 100:.2f}%" for s, d in sells))
    return out


def rebalance_hover(rec: dict, fill_ts: pd.Timestamp | None) -> str:
    """The 'why' behind one marker, as multi-line hover HTML."""
    as_of = rec.get("as_of", "?")
    if "error" in rec:
        return (f"<b>Rebalance {as_of} — FAILED</b><br>"
                f"{str(rec.get('error'))[:180]}")

    head = f"<b>Rebalance {as_of}</b>"
    if fill_ts is not None and pd.Timestamp(as_of).normalize() != fill_ts:
        head += f"  <i>(filled {fill_ts:%Y-%m-%d})</i>"

    tp = rec.get("trade_policy") or {}
    diag = rec.get("diagnostics") or {}
    # trade_policy postdates the first few runs (added 2026-07-13), so fall back to the
    # sizer's own gross rather than showing an em-dash for perfectly good history.
    gross = tp.get("gross", diag.get("gross"))
    bits = [f"{rec.get('n_orders', 0)} orders"]
    if tp.get("turnover") is not None:
        bits.append(f"turnover {_fmt_pct(tp['turnover'])}")
    if gross is not None:
        bits.append(f"gross {_fmt_pct(gross)} of equity")
    line2 = " · ".join(bits)
    errs = rec.get("n_order_errors") or 0
    if errs:
        line2 += f" · <b>{errs} order errors</b>"

    lines = [head, line2, _regime_line(rec)]
    if rec.get("kill_switch"):
        lines.append("<b>KILL SWITCH LATCHED — no orders sent</b>")
    sig = _top_signals(rec)
    if sig:
        lines.append(sig)
    lines.extend(_biggest_trades(rec))
    if rec.get("_retried_after"):
        lines.append(f"<i>recovered after a failed attempt: {rec['_retried_after']}</i>")
    return "<br>".join(lines)


def _marker_color(rec: dict) -> str:
    if "error" in rec:
        return _MARKER_FAIL
    if rec.get("kill_switch") or (rec.get("n_order_errors") or 0):
        return _MARKER_FAIL
    diag = rec.get("diagnostics") or {}
    rs = diag.get("regime_state") or {}
    if not rs or "error" in rs or rs.get("degraded", bool(rs.get("fetch_errors"))):
        return _MARKER_WARN
    return _MARKER_OK


def collapse_by_date(records: list[dict]) -> list[dict]:
    """One marker per rebalance date, with retries folded into it.

    2026-07-03 wrote both a `-failed.json` (the HON not-fractionable abort) and a
    successful recovery record. Plotted separately they land on the identical point and
    silently hide each other; the recovery is the outcome, the failure is context.
    """
    by_date: dict[str, list[dict]] = {}
    for r in records:
        by_date.setdefault(str(r.get("as_of")), []).append(r)

    out = []
    for as_of in sorted(by_date):
        group = by_date[as_of]
        ok = [r for r in group if "error" not in r]
        failed = [r for r in group if "error" in r]
        if not ok:
            out.append(group[0])
            continue
        primary = dict(ok[-1])
        if failed:
            primary["_retried_after"] = str(failed[0].get("error"))[:140]
        out.append(primary)
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def rebase(bench: pd.DataFrame, symbol: str, dates, base_value: float) -> pd.DataFrame:
    """One benchmark expressed as 'what $base_value would have become'.

    Reindexed onto the account's OWN date grid (forward-filled over any bar the account
    curve doesn't have) and anchored at its first date. Both lines then start from the
    same dollar on the same day and end on the same day — otherwise the benchmark quietly
    gets a longer window than the strategy and the comparison is not like-for-like.
    """
    idx = pd.DatetimeIndex(pd.Series(dates).sort_values().unique())
    s = bench[bench["symbol"] == symbol].sort_values("ts")
    if s.empty or idx.empty:
        return pd.DataFrame(columns=["ts", "value"])
    closes = (s.set_index("ts")["close"]
               .reindex(s.set_index("ts").index.union(idx))
               .sort_index().ffill().reindex(idx).dropna())
    if closes.empty or float(closes.iloc[0]) <= 0:
        return pd.DataFrame(columns=["ts", "value"])
    p0 = float(closes.iloc[0])
    return pd.DataFrame({"ts": closes.index,
                         "value": base_value * closes.to_numpy() / p0})


def _fill_ts(as_of, equity: pd.DataFrame) -> pd.Timestamp | None:
    """Where a rebalance lands on the equity curve.

    Friday's decision fills at Monday's open, so a marker belongs on the first session
    at or after as_of — dropping it on the decision date would point at a book that had
    not yet changed.
    """
    if equity.empty:
        return None
    later = equity[equity["ts"] >= pd.Timestamp(as_of).normalize()]
    if len(later):
        return pd.Timestamp(later["ts"].iloc[0])
    return pd.Timestamp(equity["ts"].iloc[-1])


def build_hero(
    equity: pd.DataFrame,
    benchmarks: pd.DataFrame,
    exposure: pd.DataFrame,
    sentiment: pd.DataFrame,
    records: list[dict],
    height: int = 720,
) -> go.Figure | None:
    """Assemble the unified figure. Returns None when there is nothing to draw."""
    equity = equity.dropna().sort_values("ts") if len(equity) else equity
    has_equity = len(equity) >= 2
    has_exposure = len(exposure) >= 1 and has_equity
    has_sentiment = len(sentiment) >= 2
    if not has_equity and not has_sentiment:
        return None

    specs = [("equity", 0.58)] if has_equity else []
    if has_exposure:
        specs.append(("exposure", 0.18))
    if has_sentiment:
        specs.append(("sentiment", 0.24))
    total = sum(h for _, h in specs)
    rows = {name: i + 1 for i, (name, _) in enumerate(specs)}
    titles = {"equity": "Growth of the account", "exposure": "Gross exposure",
              "sentiment": "Sentiment of the book"}

    fig = make_subplots(
        rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[h / total for _, h in specs],
        subplot_titles=[titles[n] for n, _ in specs],
    )

    # ---------------------------------------------------------------- row 1
    if has_equity:
        r = rows["equity"]
        base = float(equity["equity"].iloc[0])

        for sym, style in BENCH_STYLE.items():
            reb = rebase(benchmarks, sym, equity["ts"], base) if len(benchmarks) else None
            if reb is None or reb.empty:
                continue
            fig.add_trace(go.Scatter(
                x=reb["ts"], y=reb["value"], mode="lines", name=style["label"],
                line=dict(color=style["color"], width=1.4, dash=style["dash"]),
                legendrank=style["rank"],
                hovertemplate=f"{style['label']}<br>$%{{y:,.0f}}<extra></extra>",
            ), row=r, col=1)

        # Drawn after the benchmarks so it sits on top; legendrank puts it first in the
        # legend regardless of draw order.
        up = float(equity["equity"].iloc[-1]) >= base
        fig.add_trace(go.Scatter(
            x=equity["ts"], y=equity["equity"], mode="lines", name="QTS book",
            line=dict(color=ui.GREEN if up else ui.RED, width=2.8), legendrank=1,
            hovertemplate="QTS book<br>%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
        ), row=r, col=1)

        # rebalance markers, on the equity line at the session they took effect
        mx, my, mtext, mcolor = [], [], [], []
        for rec in collapse_by_date(records):
            ts = _fill_ts(rec.get("as_of"), equity)
            if ts is None:
                continue
            row = equity[equity["ts"] == ts]
            if row.empty:
                continue
            mx.append(ts)
            my.append(float(row["equity"].iloc[0]))
            mtext.append(rebalance_hover(rec, ts))
            mcolor.append(_marker_color(rec))
        if mx:
            fig.add_trace(go.Scatter(
                x=mx, y=my, mode="markers", name="Rebalance", legendrank=2,
                # Pale ring, not a dark one: a green "ok" marker sitting on the green
                # equity line is invisible without it.
                marker=dict(symbol="triangle-up", size=14, color=mcolor,
                            line=dict(color=ui.TEXT, width=1.6)),
                hovertext=mtext, hoverinfo="text",
            ), row=r, col=1)

        fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=r, col=1)

    # ---------------------------------------------------------------- row 2
    if has_exposure:
        r = rows["exposure"]
        exp = exposure.sort_values("as_of").copy()
        # Carry the last rebalance's exposure forward to the right edge of the whole
        # figure -- the book did not stop being invested on the most recent Friday, and
        # a step that stops short reads as "we went flat".
        right = max([pd.Timestamp(equity["ts"].iloc[-1])]
                    + ([pd.Timestamp(sentiment["dt"].max())] if has_sentiment else []))
        xs = list(exp["as_of"]) + ([right] if right > exp["as_of"].iloc[-1] else [])
        ys = list(exp["gross"]) + ([float(exp["gross"].iloc[-1])]
                                   if len(xs) > len(exp) else [])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="Invested", legendrank=6,
            line=dict(color=ui.ACCENT, width=2, shape="hv"),
            fill="tozeroy", fillcolor="rgba(91,141,239,0.16)",
            hovertemplate="Invested %{y:.0%} of equity<extra></extra>",
        ), row=r, col=1)
        fig.add_hline(y=1.0, line=dict(color=ui.MUTED, width=1, dash="dot"),
                      row=r, col=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08], row=r, col=1)

    # ---------------------------------------------------------------- row 3
    if has_sentiment:
        r = rows["sentiment"]
        sen = sentiment.sort_values("dt")
        series = [
            ("news", "Book — news", ui.AMBER, "solid", 2.2, 7),
            ("universe_news", "S&P 500 — news", ui.MUTED, "dot", 1.3, 8),
            ("reddit", "Book — Reddit", "#B48EF5", "dash", 1.6, 9),
        ]
        for col, label, color, dash, width, rank in series:
            if col not in sen.columns or sen[col].notna().sum() < 2:
                continue
            fig.add_trace(go.Scatter(
                x=sen["dt"], y=sen[col], mode="lines", name=label, legendrank=rank,
                line=dict(color=color, width=width, dash=dash),
                connectgaps=False,          # a gap is missing data, not a flat opinion
                hovertemplate=f"{label}<br>%{{y:+.3f}}<extra></extra>",
            ), row=r, col=1)
        fig.add_hline(y=0.0, line=dict(color=ui.BORDER, width=1), row=r, col=1)
        fig.update_yaxes(tickformat="+.2f", row=r, col=1)

    fig.update_layout(
        height=height,
        hovermode="x unified" if not has_equity else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        # Left margin has to clear "$100,000" -- the inherited l=10 clipped every
        # dollar tick on this figure.
        margin=dict(l=62, r=16, t=76, b=16),
    )
    for ann in fig.layout.annotations:          # subplot titles -> quiet eyebrow style
        ann.font.size = 11
        ann.font.color = ui.MUTED
        ann.xanchor = "left"
        ann.x = 0
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor=ui.BORDER, spikedash="dot")
    return fig
