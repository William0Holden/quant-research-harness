"""SEC EDGAR ingestion (key-free): XBRL fundamentals + filing events (8-K, Form 4).

All facts are stamped with knowledge_time = filing date (when the public could know it).
Fundamentals are keyed by (concept, period_end, filing_date) so restatements are distinct
rows and the as-of accessor returns only what was filed by `as_of`.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
import requests

from ..config import SEC_RATE_SLEEP_S, SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Curated us-gaap concepts that feed the v1 factor set.
CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "NetIncomeLoss",
    "StockholdersEquity",
    "Assets",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]


def _get(url: str) -> dict | None:
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    time.sleep(SEC_RATE_SLEEP_S)
    return r.json()


def fetch_fundamentals(symbol: str, cik: str) -> pd.DataFrame:
    data = _get(FACTS_URL.format(cik=cik))
    if not data:
        return pd.DataFrame()
    rows = []
    facts = data.get("facts", {}).get("us-gaap", {})
    for concept in CONCEPTS:
        node = facts.get(concept)
        if not node:
            continue
        for unit, items in node.get("units", {}).items():
            if not unit.startswith("USD") and unit != "shares":
                continue
            for it in items:
                end = it.get("end")
                filed = it.get("filed")
                val = it.get("val")
                if end is None or filed is None or val is None:
                    continue
                rows.append({
                    "symbol": symbol, "cik": cik, "concept": concept,
                    "period_end": end, "value": float(val),
                    "filing_date": filed, "form": it.get("form"),
                    "fy": it.get("fy"), "fp": it.get("fp"),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df["knowledge_time"] = pd.to_datetime(df["filing_date"])
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    # de-dup exact repeats across filings (same concept/period/filing)
    df = df.drop_duplicates(["symbol", "concept", "period_end", "filing_date"])
    return df


def fetch_events(symbol: str, cik: str, forms=("8-K", "4")) -> pd.DataFrame:
    """Filing events from the submissions feed (recent ~1000 filings).

    8-K rows carry item codes; Form-4 rows are insider-activity events. Direction
    (purchase vs sale) is a documented follow-on parse of the Form-4 XML; v1 records the
    event and lets the insider provider use filing intensity as a coarse proxy.
    """
    data = _get(SUBS_URL.format(cik=cik))
    if not data:
        return pd.DataFrame()
    recent = data.get("filings", {}).get("recent", {})
    n = len(recent.get("accessionNumber", []))
    rows = []
    for i in range(n):
        form = recent["form"][i]
        if form not in forms:
            continue
        filed = recent["filingDate"][i]
        items = recent.get("items", [""] * n)[i] or ""
        # one row per 8-K item code (or a single blank-item row otherwise)
        item_list = [c.strip() for c in items.split(",") if c.strip()] or [""]
        for item in item_list:
            rows.append({
                "symbol": symbol, "cik": cik, "form": form, "item": item,
                "filing_date": filed, "accession": recent["accessionNumber"][i],
                "extra": json.dumps({"primaryDoc": recent.get("primaryDocument", [""] * n)[i]}),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df["knowledge_time"] = pd.to_datetime(df["filing_date"])
    return df


def ingest_sec(store, members: pd.DataFrame) -> dict[str, dict]:
    """members: DataFrame with symbol, cik columns."""
    out: dict[str, dict] = {}
    for _, m in members.iterrows():
        sym, cik = m["symbol"], m["cik"]
        if not cik:
            out[sym] = {"funda": -1, "events": -1}
            continue
        rec = {}
        try:
            rec["funda"] = store.write_fundamentals(fetch_fundamentals(sym, cik))
        except Exception as e:  # noqa: BLE001
            rec["funda"] = -1
            print(f"  ! funda {sym}: {e}")
        try:
            rec["events"] = store.write_events(fetch_events(sym, cik))
        except Exception as e:  # noqa: BLE001
            rec["events"] = -1
            print(f"  ! events {sym}: {e}")
        out[sym] = rec
    return out
