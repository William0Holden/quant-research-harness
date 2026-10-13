"""Crypto text -> coin linker. Cashtag + curated alias ONLY. NO bare-ticker matching.

The equity linker matches bare uppercase tickers, which is fatal in crypto: LINK, DOT, UNI,
BAT, SKY, HYPE, ARB, POL and TRUMP are all real tickers AND common English words. A false
link is worse here than in the equity book, because mention VOLUME is a direct input to two
of the three signals -- contamination doesn't just add noise, it moves the signal. So we take
precision over recall: a coin is linked only by its cashtag ($btc) or a curated, word-bounded
name/abbreviation alias. A missed mention is cheap; a false one is not.
"""
from __future__ import annotations

import re

from ..universe import BTC, ETH

# Canonical coin -> the alias strings that map to it. Word-bounded, case-insensitive.
# Deliberately conservative: only unambiguous names/abbreviations. "eth" is safe (not an
# English word); "btc"/"eth" also appear as cashtags handled separately.
ALIASES: dict[str, tuple[str, ...]] = {
    BTC: ("bitcoin", "btc"),
    ETH: ("ethereum", "ether", "eth"),
}

# Cashtags we honour, mapped to canonical symbols. $btc / $eth, any case.
_CASHTAGS: dict[str, str] = {"btc": BTC, "xbt": BTC, "eth": ETH}

# Precompiled word-bounded alias patterns, longest-first so "ethereum" wins over "eth".
_ALIAS_PATTERNS: list[tuple[re.Pattern, str]] = sorted(
    ((re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE), sym)
     for sym, aliases in ALIASES.items() for a in aliases),
    key=lambda t: -len(t[0].pattern),
)
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{2,5})\b")


def link_text(text: str) -> set[str]:
    """Return the set of canonical coin symbols mentioned in `text`.

    Only cashtags and curated aliases match; a bare uppercase 'LINK' or 'TRUMP' never does.
    """
    if not text:
        return set()
    found: set[str] = set()
    for m in _CASHTAG_RE.finditer(text):
        sym = _CASHTAGS.get(m.group(1).lower())
        if sym:
            found.add(sym)
    for pat, sym in _ALIAS_PATTERNS:
        if pat.search(text):
            found.add(sym)
    return found


def link_symbols(text: str) -> str:
    """Comma-joined linked symbols for the corpus `symbols` column ('' = none), matching the
    equity collectors' storage convention so `aggregate` and `daily_symbol_*` reuse unchanged."""
    return ",".join(sorted(link_text(text)))
