"""Corpus -> PIT distillation: per-symbol daily item counts AND daily sentiment
(sum, n) into qts.duckdb's alt_signals table. WRITES THE MAIN DB — must run under the outputs/.qts-db.lock
flock (it is invoked from deploy/daily-pipeline.sh, which holds it), plus the
corpus flock for the read side when run via CLI."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..pit.store import PITStore
from .corpus import CorpusStore

# concept -> corpus tables summed into it (comments count as reddit mentions)
COUNT_SOURCES = {"news_count": ("news_items",),
                 "reddit_count": ("social_posts", "social_comments")}
SENT_SOURCES = {"news_sent": ("news_items",),
                "reddit_sent": ("social_posts", "social_comments")}


def _dtkey(x) -> str:
    """Normalize a dt value (datetime.date or pandas/duckdb Timestamp) to a plain
    YYYY-MM-DD string so values from the DataFrame side and the DuckDB side compare
    equal regardless of which concrete type each happened to come back as."""
    return str(pd.Timestamp(x).date())


def _write_changed(store: PITStore, concept: str, df: pd.DataFrame,
                   start: date, kt: pd.Timestamp) -> int:
    """Bitemporal integrity: re-upserting the trailing window every run must not
    restamp knowledge_time=now on rows whose value hasn't actually changed --
    that would drift kt to dt+days_back and manufacture phantom retro anomalies.
    Only write rows that are NEW (no existing row for that symbol/dt) or whose
    value CHANGED since the last aggregation; everything else keeps its old
    knowledge_time untouched by simply not being re-written."""
    existing = store.con.execute(
        "SELECT symbol, dt, value FROM alt_signals WHERE concept = ? AND dt >= ?",
        [concept, start]).df()
    existing_map = {(r.symbol, _dtkey(r.dt)): r.value for r in existing.itertuples()}
    changed = df[df.apply(
        lambda r: existing_map.get((r["symbol"], _dtkey(r["dt"]))) != r["value"],
        axis=1)].copy()
    if changed.empty:
        return 0
    changed["concept"] = concept
    changed["knowledge_time"] = kt
    return store.write_alt_signals(
        changed[["symbol", "concept", "dt", "value", "knowledge_time"]])


def run_aggregation(corpus: CorpusStore, store: PITStore, days_back: int = 14,
                    today: date | None = None) -> dict[str, int]:
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    kt = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    out: dict[str, int] = {}
    for concept, tables in COUNT_SOURCES.items():
        parts = [corpus.daily_symbol_counts(t, start, today) for t in tables]
        counts = pd.concat(parts, ignore_index=True)
        if counts.empty:
            out[concept] = 0
            continue
        df = (counts.groupby(["symbol", "dt"])["n"].sum()
                    .rename("value").reset_index())
        # daily_symbol_counts returns dt as a string (cast for test friendliness);
        # convert explicitly to date so it lands cleanly in the DATE column.
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
        df["value"] = df["value"].astype(float)
        out[concept] = _write_changed(store, concept, df, start, kt)
    for prefix, tables in SENT_SOURCES.items():
        parts = [corpus.daily_symbol_sentiment(t, start, today) for t in tables]
        sent = pd.concat(parts, ignore_index=True)
        if sent.empty:
            out[prefix] = 0
            continue
        sent = (sent.groupby(["symbol", "dt"])[["sent_sum", "sent_n"]]
                    .sum().reset_index())
        sent["dt"] = pd.to_datetime(sent["dt"]).dt.date
        n = 0
        for col, suffix in (("sent_sum", "_sum"), ("sent_n", "_n")):
            df = sent[["symbol", "dt", col]].rename(columns={col: "value"})
            df["value"] = df["value"].astype(float)
            n += _write_changed(store, prefix + suffix, df, start, kt)
        out[prefix] = n
    print(f"aggregate: {out}")
    return out


def main() -> None:
    with CorpusStore(read_only=True) as corpus, PITStore() as store:
        run_aggregation(corpus, store)


if __name__ == "__main__":
    main()
