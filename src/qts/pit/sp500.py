"""S&P 500 universe builder.

Constituents from Wikipedia's 'List of S&P 500 companies' (key-free) with a
vendored CSV fallback (sp500_fallback.csv, refreshed whenever the fetch runs).
Same survivorship caveat as the seed: rows are CURRENT constituents opened at
START; historically delisted names still enter only via DELISTED_SEED /
future Form-25 reconstruction. Symbols use Yahoo dash form (BRK-B).
"""
from __future__ import annotations

import csv
import io

import pandas as pd
import requests

from ..config import SEC_USER_AGENT
from .store import PITStore
from .universe import DELISTED_SEED, START, _cik_map

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK = __file__.rsplit("/", 1)[0] + "/sp500_fallback.csv"


def parse_sp500_html(html: str) -> list[dict]:
    tables = pd.read_html(io.StringIO(html))
    t = next(t for t in tables if "Symbol" in t.columns)
    return [
        {"symbol": str(r["Symbol"]).strip().upper().replace(".", "-"),
         "name": str(r.get("Security", r["Symbol"])).strip()}
        for _, r in t.iterrows()
    ]


def fetch_sp500() -> list[dict]:
    r = requests.get(WIKI_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    r.raise_for_status()
    rows = parse_sp500_html(r.text)
    if len(rows) < 480:
        raise ValueError(f"suspiciously small constituent table: {len(rows)}")
    return rows


def load_fallback() -> list[dict]:
    with open(FALLBACK, newline="") as f:
        return [{"symbol": r["symbol"], "name": r["name"]} for r in csv.DictReader(f)]


def _save_fallback(rows: list[dict]) -> None:
    with open(FALLBACK, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "name"])
        w.writeheader()
        w.writerows(rows)


def build_sp500_universe(store: PITStore) -> int:
    """Fetch (or fall back) + resolve CIKs + upsert. Idempotent."""
    try:
        rows = fetch_sp500()
        _save_fallback(rows)
    except Exception as e:  # noqa: BLE001
        print(f"  ! wikipedia fetch failed ({e}); using vendored fallback")
        rows = load_fallback()
    cik = _cik_map()
    recs = [{
        "symbol": r["symbol"], "cik": cik.get(r["symbol"]), "name": r["name"],
        "active_from": START, "active_to": None, "delisted": False,
    } for r in rows]
    recs.extend(DELISTED_SEED)
    df = pd.DataFrame(recs)
    df["active_from"] = pd.to_datetime(df["active_from"]).dt.date
    df["active_to"] = pd.to_datetime(df["active_to"]).dt.date
    return store.write_universe(df)
