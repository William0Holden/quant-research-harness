"""EDGAR filing-text ingestion: fetch the primary document, strip HTML, extract Risk
Factors (Item 1A) and MD&A (Item 7) from 10-K/10-Q filings, and pull 8-K earnings
press releases (Exhibit-99 of item-2.02 8-Ks) -- all stored as sections in the PIT
`filings` table.

These sections feed the NLP factory: the Lazy-Prices diff (cosine distance between
consecutive same-section filings) and FinBERT/LLM sentiment. knowledge_time = filing_date.
"""
from __future__ import annotations

import re
import time
from datetime import datetime

import pandas as pd
import requests

from ..config import SEC_RATE_SLEEP_S, SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}
SUBS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
INDEX_JSON = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/index.json"
MAX_SECTION_CHARS = 40_000


def list_filings(
    cik: str, forms: tuple[str, ...] = ("10-K",), limit_per_form: int = 6,
) -> list[dict]:
    """Most-recent filings per requested form from the submissions feed."""
    r = requests.get(SUBS_URL.format(cik=cik), headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    time.sleep(SEC_RATE_SLEEP_S)
    rec = r.json().get("filings", {}).get("recent", {})
    n = len(rec.get("accessionNumber", []))
    counts = {f: 0 for f in forms}
    out = []
    for i in range(n):
        form = rec["form"][i]
        if form not in counts or counts[form] >= limit_per_form:
            continue
        counts[form] += 1
        out.append({
            "form": form,
            "accession": rec["accessionNumber"][i],
            "doc": rec["primaryDocument"][i],
            "filing_date": rec["filingDate"][i],
            "period_end": rec.get("reportDate", [""] * n)[i] or None,
            "items": rec.get("items", [""] * n)[i] or "",
        })
        if all(c >= limit_per_form for c in counts.values()):
            break
    return out


def existing_accessions(store, symbol: str) -> set[str]:
    """Accessions already ingested for a symbol — makes re-runs cheap/resumable."""
    df = store.con.execute(
        "SELECT DISTINCT accession FROM filings WHERE symbol = ?", [symbol]).df()
    return set(df["accession"])


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#160;", " ").replace("&#8217;", "'").replace("&rsquo;", "'"))
    html = re.sub(r"&#?\w+;", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def fetch_document_text(cik: str, accession: str, doc: str) -> str:
    url = ARCHIVE.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""), doc=doc)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    time.sleep(SEC_RATE_SLEEP_S)
    return _strip_html(r.text)


def _extract(text: str, start_pat: str, end_pats: list[str]) -> str | None:
    """Pick the start-marker occurrence that yields the longest substantive section."""
    starts = [m.start() for m in re.finditer(start_pat, text, re.IGNORECASE)]
    best = None
    for s in starts:
        ends = [m.start() for p in end_pats
                for m in re.finditer(p, text[s + 10:], re.IGNORECASE)]
        e = (s + 10 + min(ends)) if ends else len(text)
        chunk = text[s:e].strip()
        if best is None or len(chunk) > len(best):
            best = chunk
    if best and len(best) > 500:  # discard TOC-only stubs
        return best[:MAX_SECTION_CHARS]
    return None


def extract_sections(text: str, form: str = "10-K") -> dict[str, str]:
    out: dict[str, str] = {}
    if form == "10-Q":
        rf = _extract(text, r"item\s*1a\.?\s*risk\s*factors",
                      [r"item\s*2\b", r"item\s*5\b", r"item\s*6\b"])
        mda = _extract(text, r"item\s*2\.?\s*management.?s\s*discussion",
                       [r"item\s*3\b", r"item\s*4\b"])
    else:
        rf = _extract(text, r"item\s*1a\.?\s*risk\s*factors",
                      [r"item\s*1b", r"item\s*2\b", r"item\s*7\b"])
        mda = _extract(text, r"item\s*7\.?\s*management.?s\s*discussion",
                       [r"item\s*7a", r"item\s*8\b"])
    if rf:
        out["risk_factors"] = rf
    if mda:
        out["mda"] = mda
    # Always keep a truncated full-document section as a guaranteed signal carrier
    # (section extraction is heuristic and misses some layouts, esp. financials).
    out["full"] = text[:MAX_SECTION_CHARS]
    return out


def ingest_filings(store, members: pd.DataFrame, forms: tuple[str, ...] = ("10-K",), n_per_symbol: int = 6) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, m in members.iterrows():
        sym, cik = m["symbol"], m["cik"]
        if not cik:
            out[sym] = 0
            continue
        rows = []
        seen = existing_accessions(store, sym)
        try:
            for f in list_filings(cik, forms, n_per_symbol):
                if f["accession"] in seen:
                    continue
                try:
                    txt = fetch_document_text(cik, f["accession"], f["doc"])
                except Exception:
                    continue
                for section, body in extract_sections(txt, form=f["form"]).items():
                    rows.append({
                        "symbol": sym, "cik": cik, "form": f["form"],
                        "period_end": f["period_end"], "filing_date": f["filing_date"],
                        "accession": f["accession"], "section": section, "text": body,
                        "knowledge_time": f["filing_date"],
                    })
        except Exception as e:  # noqa: BLE001
            print(f"  ! filings {sym}: {e}")
        if rows:
            df = pd.DataFrame(rows)
            df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce").dt.date
            df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
            df["knowledge_time"] = pd.to_datetime(df["knowledge_time"])
            out[sym] = store.write_filings(df)
        else:
            out[sym] = 0
        print(f"  filings {sym}: {out[sym]} sections")
    return out


def find_ex99_doc(cik: str, accession: str) -> str | None:
    """Locate the Exhibit-99 (earnings press release) doc in a filing's index."""
    url = INDEX_JSON.format(cik_int=int(cik), acc_nodash=accession.replace("-", ""))
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    time.sleep(SEC_RATE_SLEEP_S)
    items = r.json().get("directory", {}).get("item", [])
    cands = [it["name"] for it in items
             if "99" in it["name"] and it["name"].lower().endswith((".htm", ".html"))]
    return min(cands, key=len) if cands else None


def ingest_press_releases(store, members: pd.DataFrame, n_per_symbol: int = 8) -> dict[str, int]:
    """Earnings press releases: Ex-99 text of item-2.02 8-Ks -> filings table
    (form='8-K', section='press_release'). Resumable via existing_accessions."""
    out: dict[str, int] = {}
    for _, m in members.iterrows():
        sym, cik = m["symbol"], m["cik"]
        if not cik:
            out[sym] = 0
            continue
        rows = []
        try:
            seen = existing_accessions(store, sym)
            eightks = [f for f in list_filings(cik, forms=("8-K",), limit_per_form=40)
                       if "2.02" in f["items"]][:n_per_symbol]
            for f in eightks:
                if f["accession"] in seen:
                    continue
                doc = find_ex99_doc(cik, f["accession"])
                if not doc:
                    continue
                try:
                    txt = fetch_document_text(cik, f["accession"], doc)
                except Exception:
                    continue
                if len(txt) < 500:
                    continue
                rows.append({
                    "symbol": sym, "cik": cik, "form": "8-K",
                    "period_end": f["period_end"], "filing_date": f["filing_date"],
                    "accession": f["accession"], "section": "press_release",
                    "text": txt[:MAX_SECTION_CHARS],
                    "knowledge_time": f["filing_date"],
                })
        except Exception as e:  # noqa: BLE001
            print(f"  ! press_releases {sym}: {e}")
        if rows:
            df = pd.DataFrame(rows)
            df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce").dt.date
            df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
            df["knowledge_time"] = pd.to_datetime(df["knowledge_time"])
            out[sym] = store.write_filings(df)
        else:
            out[sym] = 0
        print(f"  press_release {sym}: {out[sym]}")
    return out


if __name__ == "__main__":
    import datetime as _d

    from ..pit.store import PITStore

    _s = PITStore()
    try:
        _m = _s.universe_as_of(_d.date.today())
        _m = _m[_m["cik"].notna()]
        ingest_filings(_s, _m, forms=("10-K", "10-Q"), n_per_symbol=8)
        ingest_press_releases(_s, _m)
    finally:
        _s.close()
