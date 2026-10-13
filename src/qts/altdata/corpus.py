"""Raw alt-data corpus — SEPARATE DuckDB from the PIT lake.

Collectors (news, reddit) write here hourly; the main qts.duckdb is never touched
by collectors. All access is serialized by flock on outputs/.qts-corpus.lock at the
shell level (deploy/*.sh). knowledge_time convention: fetched_at (crawl time) —
claimed publish times are stored but never trusted for PIT.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from ..config import CORPUS_DB_PATH, ensure_dirs

CORPUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_items (
    url_hash             VARCHAR PRIMARY KEY,
    url                  VARCHAR,
    source               VARCHAR,
    title                VARCHAR,
    summary              VARCHAR,
    body                 VARCHAR,
    claimed_published_at TIMESTAMP,
    fetched_at           TIMESTAMP NOT NULL,   -- knowledge_time basis
    symbols              VARCHAR,               -- comma-joined linked tickers ('' = none)
    title_hash           VARCHAR,
    extract_ok           BOOLEAN
);
CREATE TABLE IF NOT EXISTS social_posts (
    post_id      VARCHAR PRIMARY KEY,
    platform     VARCHAR,
    subreddit    VARCHAR,
    author_hash  VARCHAR,
    title        VARCHAR,
    body         VARCHAR,
    score        INTEGER,
    num_comments INTEGER,
    created_at   TIMESTAMP,
    fetched_at   TIMESTAMP NOT NULL,
    symbols      VARCHAR
);
CREATE TABLE IF NOT EXISTS social_post_obs (
    post_id      VARCHAR,
    fetched_at   TIMESTAMP NOT NULL,
    score        INTEGER,
    num_comments INTEGER,
    PRIMARY KEY (post_id, fetched_at)
);
CREATE TABLE IF NOT EXISTS social_comments (
    comment_id  VARCHAR PRIMARY KEY,
    post_id     VARCHAR,
    subreddit   VARCHAR,
    author_hash VARCHAR,
    body        VARCHAR,
    score       INTEGER,
    created_at  TIMESTAMP,
    fetched_at  TIMESTAMP NOT NULL,
    symbols     VARCHAR
);
CREATE TABLE IF NOT EXISTS comment_fetch_state (
    post_id           VARCHAR PRIMARY KEY,
    last_num_comments INTEGER,
    last_fetched_at   TIMESTAMP
);
CREATE TABLE IF NOT EXISTS text_sentiment (
    text_id    VARCHAR PRIMARY KEY,   -- 'post:<id>' | 'news:<url_hash>' | 'comment:<id>'
    value      DOUBLE,                -- [-1, 1], bearish..bullish
    model      VARCHAR,
    scored_at  TIMESTAMP NOT NULL     -- PIT knowledge_time basis for sentiment
);
"""


class CorpusStore:
    def __init__(self, path: Path | str = CORPUS_DB_PATH, read_only: bool = False):
        ensure_dirs()
        self.path = str(path)
        self.con = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self.con.execute(CORPUS_SCHEMA)

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.con.close()

    def close(self) -> None:
        self.con.close()

    def _upsert(self, table: str, df: pd.DataFrame, pk: str | list[str]) -> int:
        if df is None or df.empty:
            return 0
        pks = [pk] if isinstance(pk, str) else list(pk)
        self.con.register("_staging", df)
        cols = ", ".join(df.columns)
        cond = " AND ".join(f"t.{c} = s.{c}" for c in pks)
        self.con.execute(f"DELETE FROM {table} t USING _staging s WHERE {cond}")
        self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _staging")
        self.con.unregister("_staging")
        return len(df)

    def upsert_news(self, df: pd.DataFrame) -> int:
        return self._upsert("news_items", df, "url_hash")

    def upsert_posts(self, df: pd.DataFrame) -> int:
        return self._upsert("social_posts", df, "post_id")

    def upsert_post_obs(self, df: pd.DataFrame) -> int:
        return self._upsert("social_post_obs", df, ["post_id", "fetched_at"])

    def upsert_comments(self, df: pd.DataFrame) -> int:
        return self._upsert("social_comments", df, "comment_id")

    def upsert_comment_state(self, df: pd.DataFrame) -> int:
        return self._upsert("comment_fetch_state", df, "post_id")

    def comment_fetch_state(self, post_ids: list[str]) -> dict[str, int]:
        """post_id -> num_comments at last comment fetch (missing = never fetched)."""
        if not post_ids:
            return {}
        ph = ", ".join(["?"] * len(post_ids))
        df = self.con.execute(
            f"SELECT post_id, last_num_comments FROM comment_fetch_state "
            f"WHERE post_id IN ({ph})", post_ids).df()
        return dict(zip(df["post_id"], df["last_num_comments"].astype(int)))

    def upsert_sentiment(self, df: pd.DataFrame) -> int:
        return self._upsert("text_sentiment", df, "text_id")

    def unscored_texts(self, limit: int = 2000) -> pd.DataFrame:
        """Newest-first symbol-linked texts with no sentiment row yet -> [text_id, text]."""
        q = """
            WITH cand AS (
              SELECT 'news:' || url_hash AS text_id,
                     COALESCE(title, '') || '. ' || COALESCE(body, summary, '') AS text,
                     fetched_at
              FROM news_items WHERE symbols <> ''
              UNION ALL
              SELECT 'post:' || post_id,
                     COALESCE(title, '') || '. ' || COALESCE(body, ''),
                     fetched_at
              FROM social_posts WHERE symbols <> ''
              UNION ALL
              SELECT 'comment:' || comment_id, COALESCE(body, ''), fetched_at
              FROM social_comments WHERE symbols <> ''
            )
            SELECT c.text_id, c.text FROM cand c
            LEFT JOIN text_sentiment ts ON ts.text_id = c.text_id
            WHERE ts.text_id IS NULL
            ORDER BY c.fetched_at DESC
            LIMIT ?
        """
        return self.con.execute(q, [limit]).df()

    def daily_symbol_sentiment(self, table: str, start: date, end: date) -> pd.DataFrame:
        """Per (symbol, fetch-date) sentiment sum + count over scored, linked texts."""
        prefix, idcol = {"news_items": ("news", "url_hash"),
                         "social_posts": ("post", "post_id"),
                         "social_comments": ("comment", "comment_id")}[table]
        q = (f"SELECT t.symbols, CAST(t.fetched_at AS DATE) AS dt, ts.value "
             f"FROM {table} t JOIN text_sentiment ts "
             f"ON ts.text_id = '{prefix}:' || t.{idcol} "
             f"WHERE t.fetched_at >= ? AND t.fetched_at < ? AND t.symbols <> ''")
        df = self.con.execute(
            q, [pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)]).df()
        if df.empty:
            return pd.DataFrame(columns=["symbol", "dt", "sent_sum", "sent_n"])
        df["symbol"] = df["symbols"].str.split(",")
        x = df.explode("symbol")
        out = (x.groupby(["symbol", "dt"])["value"]
                .agg(sent_sum="sum", sent_n="count").reset_index())
        out["dt"] = out["dt"].astype(str)
        return out

    def existing_hashes(self, table: str, col: str, values: list[str]) -> set[str]:
        if not values:
            return set()
        ph = ", ".join(["?"] * len(values))
        df = self.con.execute(
            f"SELECT DISTINCT {col} AS v FROM {table} WHERE {col} IN ({ph})", values).df()
        return set(df["v"])

    def counts(self) -> dict[str, int]:
        return {t: self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("news_items", "social_posts", "social_comments")}

    def daily_symbol_counts(self, table: str, start: date, end: date) -> pd.DataFrame:
        """Per (symbol, fetch-date) item counts; rows with no linked symbols excluded."""
        df = self.con.execute(
            f"SELECT symbols, CAST(fetched_at AS DATE) AS dt FROM {table} "
            f"WHERE fetched_at >= ? AND fetched_at < ? AND symbols <> ''",
            [pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)]).df()
        if df.empty:
            return pd.DataFrame(columns=["symbol", "dt", "n"])
        df["symbol"] = df["symbols"].str.split(",")
        out = (df.explode("symbol").groupby(["symbol", "dt"]).size()
                 .rename("n").reset_index())
        out["dt"] = out["dt"].astype(str)
        return out
