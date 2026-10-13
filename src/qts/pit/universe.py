"""Survivorship-free universe building blocks, shared with pit/sp500.py.

The *table* is survivorship-free (carries active_from/active_to/delisted). This module
holds the pieces sp500.py's real S&P 500 universe builder imports: a CIK resolver
against SEC's free company_tickers.json, the universe start date, and one deliberately
delisted seed name to exercise the machinery honestly.
"""
from __future__ import annotations

from datetime import date

import requests

from ..config import SEC_USER_AGENT

# A historically delisted large cap (acquired/bankrupt) to prove the table is
# survivorship-free in structure. active_to set; appears in universe_as_of() before it.
DELISTED_SEED: list[dict] = [
    # First Republic Bank — failed/seized May 2023.
    {"symbol": "FRC", "cik": None, "name": "First Republic Bank",
     "active_from": date(2010, 1, 1), "active_to": date(2023, 5, 1), "delisted": True},
]

START = date(2000, 1, 1)


def _cik_map() -> dict[str, str]:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": SEC_USER_AGENT}, timeout=30,
    )
    r.raise_for_status()
    out: dict[str, str] = {}
    for row in r.json().values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out
