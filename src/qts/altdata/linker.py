"""Text -> ticker linking. PRECISION over recall: a false link poisons the
per-symbol counts far more than a missed one. Three matchers, strictest first:
cashtags ($TSLA, any-case), bare UPPERCASE ticker tokens (stoplisted), and
multi-word company-name aliases (single-word names never match by name)."""
from __future__ import annotations

import re

import pandas as pd

# Universe tickers that are common English words/abbreviations — never matched bare.
STOP_TICKERS = {
    "A", "ALL", "AN", "ANY", "ARE", "BE", "BIG", "BY", "CAN", "COST", "DAY", "DD",
    "EAT", "FAST", "FOR", "GO", "HAS", "HE", "IT", "KEY", "LOW", "MAIN", "MET",
    "NEXT", "NOW", "ON", "ONE", "OR", "OUT", "PM", "RH", "SEE", "SO", "SEP",
    "TECH", "TWO", "UPS", "WELL", "YOU", "CEO", "ETF", "IPO", "USA", "YOLO",
}
_LEADING_THE = re.compile(r"^\s*the\s+", re.IGNORECASE)
# Trailing legal designators only, end-anchored. Deliberately excludes
# "holdings"/"group"/"the" -- those are brand words (e.g. "CBRE Group",
# "Goldman Sachs Group") and stripping them collapses names to a single
# word (dropped) or produces cores that falsely match unrelated text.
_TRAILING_SUFFIX = re.compile(
    r"[,\s]+(incorporated|inc|corporation|corp|company|co|ltd|plc|class\s+[a-c])\s*\.?\s*$",
    re.IGNORECASE)


def _clean_core(name: str) -> str:
    core = _LEADING_THE.sub("", name, count=1)
    # Apply iteratively (up to 2x) for stacked forms like "..., Inc." after
    # a first designator was already stripped (e.g. "X Corporation, Inc.").
    for _ in range(2):
        stripped = _TRAILING_SUFFIX.sub("", core)
        if stripped == core:
            break
        core = stripped
    core = re.sub(r"\s+", " ", core).strip(" ,.&").lower()
    return core


def build_alias_map(members: pd.DataFrame) -> dict[str, str]:
    """lowercase cleaned multi-word name -> symbol. Single-word cores are dropped.

    If two symbols produce the same core, the core is ambiguous and is removed
    from the map entirely (neither symbol gets a name alias) -- cashtags and
    bare tickers still cover them. Precision over recall."""
    amap: dict[str, str] = {}
    ambiguous: set[str] = set()
    for _, m in members.iterrows():
        name = str(m.get("name") or "")
        core = _clean_core(name)
        if not core or len(core.split()) < 2:
            continue
        if core in ambiguous:
            continue
        if core in amap and amap[core] != m["symbol"]:
            del amap[core]
            ambiguous.add(core)
            continue
        amap[core] = m["symbol"]
    return amap


def link_text(text: str, symbols: set[str], alias_map: dict[str, str]) -> set[str]:
    found: set[str] = set()
    # 1) cashtags (case-insensitive)
    for m in re.finditer(r"\$([A-Za-z]{1,5})\b", text):
        t = m.group(1).upper()
        if t in symbols:
            found.add(t)
    # 2) bare tickers: token must appear UPPERCASE in the original text
    for m in re.finditer(r"\b([A-Z]{2,5})\b", text):
        t = m.group(1)
        if t in symbols and t not in STOP_TICKERS:
            found.add(t)
    # 3) multi-word company names, word-bounded, case-insensitive
    low = text.lower()
    for core, sym in alias_map.items():
        if re.search(r"\b" + re.escape(core) + r"\b", low):
            found.add(sym)
    return found
