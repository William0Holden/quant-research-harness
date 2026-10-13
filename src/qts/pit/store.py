"""Bitemporal point-in-time store over DuckDB.

THE central leak guard (#1, #2). Every fact carries `knowledge_time` — when it became
knowable. The ONLY read accessors take `as_of` and filter `knowledge_time <= as_of`.
There is deliberately no `get_latest()` that ignores `as_of`.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from ..config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    symbol      VARCHAR NOT NULL,
    cik         VARCHAR,
    name        VARCHAR,
    active_from DATE    NOT NULL,
    active_to   DATE,                 -- NULL = still active
    delisted    BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (symbol, active_from)
);

CREATE TABLE IF NOT EXISTS prices (
    symbol         VARCHAR NOT NULL,
    dt             DATE    NOT NULL,
    close          DOUBLE,
    adj_close      DOUBLE,
    volume         BIGINT,
    knowledge_time TIMESTAMP NOT NULL,   -- = dt EOD; close knowable after the close
    PRIMARY KEY (symbol, dt)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol         VARCHAR NOT NULL,
    cik            VARCHAR,
    concept        VARCHAR NOT NULL,     -- XBRL tag, e.g. NetIncomeLoss
    period_end     DATE    NOT NULL,
    value          DOUBLE,
    filing_date    DATE    NOT NULL,
    form           VARCHAR,
    fy             INTEGER,
    fp             VARCHAR,
    knowledge_time TIMESTAMP NOT NULL,   -- = filing_date; keyed by filing, never period_end
    PRIMARY KEY (symbol, concept, period_end, filing_date)
);

CREATE TABLE IF NOT EXISTS events (
    symbol         VARCHAR NOT NULL,
    cik            VARCHAR,
    form           VARCHAR NOT NULL,     -- 8-K, 4, ...
    item           VARCHAR,              -- 8-K item code if present
    filing_date    DATE    NOT NULL,
    accession      VARCHAR,
    extra          VARCHAR,              -- json blob (e.g. insider txn detail)
    knowledge_time TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, form, accession, item)
);

CREATE TABLE IF NOT EXISTS filings (
    symbol         VARCHAR NOT NULL,
    cik            VARCHAR,
    form           VARCHAR,              -- 10-K / 10-Q
    period_end     DATE,
    filing_date    DATE    NOT NULL,
    accession      VARCHAR,
    section        VARCHAR,              -- 'risk_factors' | 'mda' | 'full'
    text           VARCHAR,
    knowledge_time TIMESTAMP NOT NULL,   -- = filing_date
    PRIMARY KEY (symbol, accession, section)
);

-- NLP features precomputed by the GPU worker batch job (frozen weights => PIT-honest).
CREATE TABLE IF NOT EXISTS nlp_features (
    symbol         VARCHAR NOT NULL,
    concept        VARCHAR NOT NULL,     -- 'filing_sentiment' | 'lazy_prices_diff'
    filing_date    DATE    NOT NULL,
    period_end     DATE,
    value          DOUBLE,
    knowledge_time TIMESTAMP NOT NULL,   -- = filing_date
    PRIMARY KEY (symbol, concept, filing_date)
);

-- Daily alt-data aggregates (news/social counts). knowledge_time = aggregation
-- run time (crawl-time basis: forward-collected, no backfill possible).
CREATE TABLE IF NOT EXISTS alt_signals (
    symbol         VARCHAR NOT NULL,
    concept        VARCHAR NOT NULL,     -- 'news_count' | 'reddit_count'
    dt             DATE    NOT NULL,
    value          DOUBLE,
    knowledge_time TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, concept, dt)
);
"""


class PITStore:
    def __init__(self, path: Path | str = DB_PATH, read_only: bool = False):
        ensure_dirs()
        self.path = str(path)
        self.con = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA)

    # ---- context manager -------------------------------------------------
    def __enter__(self) -> "PITStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # ---- writers (upsert via delete+insert on PK) ------------------------
    def _upsert(self, table: str, df: pd.DataFrame, pk: list[str]) -> int:
        if df is None or df.empty:
            return 0
        self.con.register("_staging", df)
        cols = ", ".join(df.columns)
        on = " AND ".join(f"t.{k} = s.{k}" for k in pk)
        # delete rows that collide on PK, then insert fresh
        self.con.execute(
            f"DELETE FROM {table} t USING _staging s WHERE {on}"
        )
        self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _staging")
        self.con.unregister("_staging")
        return len(df)

    def write_universe(self, df: pd.DataFrame) -> int:
        return self._upsert("universe", df, ["symbol", "active_from"])

    def write_prices(self, df: pd.DataFrame) -> int:
        return self._upsert("prices", df, ["symbol", "dt"])

    def write_fundamentals(self, df: pd.DataFrame) -> int:
        return self._upsert(
            "fundamentals", df, ["symbol", "concept", "period_end", "filing_date"]
        )

    def write_events(self, df: pd.DataFrame) -> int:
        return self._upsert("events", df, ["symbol", "form", "accession", "item"])

    def write_filings(self, df: pd.DataFrame) -> int:
        return self._upsert("filings", df, ["symbol", "accession", "section"])

    def write_nlp_features(self, df: pd.DataFrame) -> int:
        return self._upsert("nlp_features", df, ["symbol", "concept", "filing_date"])

    def write_alt_signals(self, df: pd.DataFrame) -> int:
        return self._upsert("alt_signals", df, ["symbol", "concept", "dt"])

    # ---- filing-corpus access (for the NLP factory) ----------------------
    def filings_export(self, section: str | None = None) -> pd.DataFrame:
        """All filing sections (for the GPU worker job to embed/score). Not as-of: the job
        computes features per filing; PIT is enforced later via filing_date knowledge_time."""
        q = "SELECT symbol, cik, form, period_end, filing_date, accession, section, text FROM filings"
        if section:
            q += " WHERE section = ?"
            return self.con.execute(q, [section]).df()
        return self.con.execute(q).df()

    def nlp_features_as_of(
        self, symbols: list[str], concept: str, as_of: datetime | date,
    ) -> pd.Series:
        """Latest known NLP feature per symbol as of `as_of` (filing_date <= as_of)."""
        as_of = _to_ts(as_of)
        q = """
            WITH known AS (
                SELECT symbol, value,
                  ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY filing_date DESC) AS rn
                FROM nlp_features
                WHERE symbol IN ({ps}) AND concept = ? AND knowledge_time <= ?
            )
            SELECT symbol, value FROM known WHERE rn = 1
        """.format(ps=_inlist(symbols))
        df = self.con.execute(q, list(symbols) + [concept, as_of]).df()
        return df.set_index("symbol")["value"].reindex(symbols) if len(df) else pd.Series(dtype=float, index=symbols)

    def alt_signals_window(
        self, symbols: list[str], concept: str, start: date, as_of: datetime | date,
    ) -> pd.DataFrame:
        """Daily alt-signal values in [start, as_of-date], knowledge_time-gated."""
        as_of = _to_ts(as_of)
        q = """
            SELECT symbol, dt, value FROM alt_signals
            WHERE symbol IN ({ps}) AND concept = ?
              AND dt >= ? AND knowledge_time <= ?
            ORDER BY symbol, dt
        """.format(ps=_inlist(symbols))
        return self.con.execute(q, list(symbols) + [concept, start, as_of]).df()

    # ---- as-of readers (leak guards) -------------------------------------
    def universe_as_of(self, as_of: date) -> pd.DataFrame:
        """Survivorship-free: names alive at `as_of`, including ones that later died."""
        return self.con.execute(
            """
            SELECT symbol, cik, name, active_from, active_to, delisted
            FROM universe
            WHERE active_from <= ?
              AND (active_to IS NULL OR active_to > ?)
            ORDER BY symbol
            """,
            [as_of, as_of],
        ).df()

    def prices_as_of(
        self, symbols: list[str], as_of: datetime | date,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """All price bars with knowledge_time <= as_of (i.e. known by then)."""
        as_of = _to_ts(as_of)
        q = """
            SELECT symbol, dt, close, adj_close, volume
            FROM prices
            WHERE symbol IN ({ph})
              AND knowledge_time <= ?
        """.format(ph=_inlist(symbols))
        params = list(symbols) + [as_of]
        if lookback_days is not None:
            q += " AND dt >= ?"
            params.append(_to_date(as_of) - pd.Timedelta(days=lookback_days))
        q += " ORDER BY symbol, dt"
        return self.con.execute(q, params).df()

    def fundamentals_as_of(
        self, symbols: list[str], concepts: list[str], as_of: datetime | date,
        annual_only: bool = False,
    ) -> pd.DataFrame:
        """Latest known value per (symbol, concept) as of `as_of`.

        'Latest known' = most recent period_end among rows already filed
        (knowledge_time <= as_of), breaking ties by latest filing_date (restatements).
        annual_only restricts to full-year facts (fp='FY') — use for flow concepts
        (income, revenue, cogs) so you get a TTM-equivalent, not a single quarter.
        """
        as_of = _to_ts(as_of)
        fp_filter = "AND fp = 'FY'" if annual_only else ""
        q = """
            WITH known AS (
                SELECT *,
                  ROW_NUMBER() OVER (
                    PARTITION BY symbol, concept
                    ORDER BY period_end DESC, filing_date DESC
                  ) AS rn
                FROM fundamentals
                WHERE symbol IN ({ps})
                  AND concept IN ({pc})
                  AND knowledge_time <= ?
                  {fp}
            )
            SELECT symbol, cik, concept, period_end, value, filing_date, form, fy, fp
            FROM known WHERE rn = 1
        """.format(ps=_inlist(symbols), pc=_inlist(concepts), fp=fp_filter)
        return self.con.execute(q, list(symbols) + list(concepts) + [as_of]).df()

    def events_between(
        self, symbols: list[str], form: str, start: date, as_of: datetime | date,
    ) -> pd.DataFrame:
        as_of = _to_ts(as_of)
        q = """
            SELECT symbol, form, item, filing_date, accession, extra
            FROM events
            WHERE symbol IN ({ps}) AND form = ?
              AND filing_date >= ? AND knowledge_time <= ?
            ORDER BY symbol, filing_date
        """.format(ps=_inlist(symbols))
        return self.con.execute(q, list(symbols) + [form, start, as_of]).df()

    # ---- introspection (for dashboard / data-status panel) ---------------
    def table_counts(self) -> dict[str, int]:
        out = {}
        for t in ("universe", "prices", "fundamentals", "events", "filings", "nlp_features",
                  "alt_signals"):
            out[t] = self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return out

    def price_coverage(self) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT symbol, COUNT(*) AS bars, MIN(dt) AS first_dt, MAX(dt) AS last_dt
            FROM prices GROUP BY symbol ORDER BY symbol
            """
        ).df()


# ---- helpers -------------------------------------------------------------
def _inlist(items: list[str]) -> str:
    return ", ".join(["?"] * len(items)) if items else "NULL"


def _to_ts(x: datetime | date) -> datetime:
    if isinstance(x, datetime):
        return x
    return datetime(x.year, x.month, x.day, 23, 59, 59)


def _to_date(x: datetime | date) -> date:
    return x.date() if isinstance(x, datetime) else x
