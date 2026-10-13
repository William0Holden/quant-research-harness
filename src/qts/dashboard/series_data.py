"""Time series that explain the equity curve: book exposure and news/Reddit sentiment.

Read-only, cached, and independently degradable — each feed returns an empty frame when
its source is missing so the hero chart can drop a row rather than fail to render.

Sentiment comes from the nightly `alt_signals` aggregation (news_sent_sum / news_sent_n
per symbol per day), not from the raw corpus: the aggregation is the same PIT table the
gated live signals read, so the chart shows what the strategy sees.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .common import get_store

NEWS_CONCEPTS = ("news_sent_sum", "news_sent_n")
REDDIT_CONCEPTS = ("reddit_sent_sum", "reddit_sent_n")


@st.cache_data(ttl=900, show_spinner=False)
def sentiment_panel(start: str) -> pd.DataFrame:
    """Per-symbol daily mean sentiment. Columns: dt, symbol, news, news_n, reddit, reddit_n.

    Rows are deduplicated to the latest knowledge_time per (symbol, concept, dt): the
    nightly aggregation rewrites a day's row when new items land, and summing across
    knowledge_times would double-count.
    """
    cols = ["dt", "symbol", "news", "news_n", "reddit", "reddit_n"]
    concepts = NEWS_CONCEPTS + REDDIT_CONCEPTS
    sql = f"""
        WITH latest AS (
            SELECT symbol, concept, dt, value,
                   ROW_NUMBER() OVER (PARTITION BY symbol, concept, dt
                                      ORDER BY knowledge_time DESC) AS rn
            FROM alt_signals
            WHERE concept IN ({','.join('?' for _ in concepts)}) AND dt >= ?
        )
        SELECT symbol, concept, dt, value FROM latest WHERE rn = 1
    """
    try:
        df = get_store().con.execute(sql, [*concepts, start]).fetchdf()
    except Exception:  # noqa: BLE001  DB locked by the writer / table absent
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)

    wide = (df.pivot_table(index=["dt", "symbol"], columns="concept",
                           values="value", aggfunc="sum")
              .reset_index())
    for c in concepts:
        if c not in wide.columns:
            wide[c] = 0.0
    wide["dt"] = pd.to_datetime(wide["dt"])
    out = pd.DataFrame({
        "dt": wide["dt"],
        "symbol": wide["symbol"],
        "news_n": wide["news_sent_n"].fillna(0.0),
        "reddit_n": wide["reddit_sent_n"].fillna(0.0),
    })
    # Mean = sum/n, undefined (NaN) rather than 0 where the day had no scored items:
    # "no news" is not "neutral news", and averaging the two together would bury signal.
    out["news"] = wide["news_sent_sum"] / wide["news_sent_n"].replace(0.0, pd.NA)
    out["reddit"] = wide["reddit_sent_sum"] / wide["reddit_sent_n"].replace(0.0, pd.NA)
    return out[cols].sort_values(["dt", "symbol"]).reset_index(drop=True)


def _held_weights_by_date(records: list[dict], dates: pd.Series) -> dict:
    """{date -> {symbol: weight}} using the book actually held on that date.

    A rebalance decided on Friday fills at Monday's open, so the weights it produced
    describe the book from the NEXT trading day onward, not the day of the decision.
    """
    stamped = []
    for r in records:
        if "error" in r or not r.get("weights"):
            continue
        try:
            stamped.append((pd.Timestamp(r["as_of"]).normalize(), r["weights"]))
        except Exception:  # noqa: BLE001
            continue
    stamped.sort(key=lambda t: t[0])
    if not stamped:
        return {}
    out = {}
    for d in pd.to_datetime(pd.Series(sorted(set(dates)))):
        prior = [w for ts, w in stamped if ts < d]
        if prior:
            out[d] = prior[-1]
    return out


def book_sentiment(records: list[dict], start: str) -> pd.DataFrame:
    """Sentiment of the names we actually hold, weighted by position size.

    Columns: dt, news, reddit, universe_news. The universe column is the same
    calculation across every scored name, for context: book above universe means the
    ranker happened to pick better-liked names, not that the market cheered up.
    """
    cols = ["dt", "news", "reddit", "universe_news"]
    panel = sentiment_panel(start)
    if panel.empty:
        return pd.DataFrame(columns=cols)

    # Universe reference: item-count-weighted, so a symbol with 60 articles counts more
    # than one with 1 — this is the market-wide mood, not a mean of per-symbol means.
    uni = panel.groupby("dt").apply(
        lambda g: (g["news"] * g["news_n"]).sum() / g["news_n"].sum()
        if g["news_n"].sum() else float("nan"),
        include_groups=False,
    ).rename("universe_news").reset_index()

    wmap = _held_weights_by_date(records, panel["dt"])
    rows = []
    for dt, g in panel.groupby("dt"):
        w = wmap.get(dt)
        if not w:
            continue
        gw = g.assign(w=g["symbol"].map(w)).dropna(subset=["w"])
        if gw.empty:
            continue

        def _wavg(col: str) -> float:
            sub = gw.dropna(subset=[col])
            tot = sub["w"].sum()
            return float((sub[col] * sub["w"]).sum() / tot) if tot > 0 else float("nan")

        rows.append({"dt": dt, "news": _wavg("news"), "reddit": _wavg("reddit")})

    book = pd.DataFrame(rows, columns=["dt", "news", "reddit"])
    out = uni.merge(book, on="dt", how="left") if not book.empty else uni.assign(
        news=float("nan"), reddit=float("nan"))
    return out[cols].sort_values("dt").reset_index(drop=True)


def exposure_series(records: list[dict]) -> pd.DataFrame:
    """Gross exposure the book actually carried out of each rebalance.

    Columns: as_of, gross, target_gross, regime_multiplier, degraded. `gross` is the
    post-trade-policy figure (what we hold); `target_gross` is what the sizer asked for.
    """
    cols = ["as_of", "gross", "target_gross", "regime_multiplier", "degraded"]
    rows = []
    for r in records:
        if "error" in r:
            continue
        tp = r.get("trade_policy") or {}
        diag = r.get("diagnostics") or {}
        rs = diag.get("regime_state") or {}
        gross = tp.get("gross", diag.get("gross"))
        if gross is None:
            continue
        rows.append({
            "as_of": pd.Timestamp(r["as_of"]).normalize(),
            "gross": float(gross),
            "target_gross": float(tp.get("target_gross", diag.get("gross") or gross)),
            "regime_multiplier": float(diag.get("regime_gross_multiplier", 1.0)),
            # A regime read that could not be evaluated. Older records predate the
            # `degraded` key, so a non-empty fetch_errors list stands in for it.
            "degraded": bool(rs.get("degraded", bool(rs.get("fetch_errors"))
                                    or "error" in rs)),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("as_of").reset_index(drop=True)
