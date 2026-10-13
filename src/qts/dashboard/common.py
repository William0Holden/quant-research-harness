"""Shared cached resources for the dashboard views (PIT store, providers, backtests)."""
from __future__ import annotations

import json
from datetime import date

import streamlit as st

from qts.backtest.engine import run_backtest
from qts.config import OUTPUTS_DIR, CostModel, live_gated
from qts.features.base import backtestable_providers, default_providers
from qts.pit.store import PITStore
from qts.research.ic import provider_ic

REBALANCE_DIR = OUTPUTS_DIR / "rebalances"
CRYPTO_REBALANCE_DIR = OUTPUTS_DIR / "rebalances_crypto"


@st.cache_resource
def get_store() -> PITStore:
    return PITStore(read_only=True)


@st.cache_resource
def get_providers():
    return default_providers()


def live_providers():
    gated = live_gated()
    return [p for p in backtestable_providers() if p.name not in gated]


@st.cache_data(show_spinner="Running walk-forward backtest…")
def cached_backtest(start: date, end: date, freq: str, n_buckets: int,
                    spread: float, slippage: float, borrow: float,
                    sizing: bool = True):
    """sizing=True simulates the book that actually trades (vol targeting, regime brake,
    caps, turnover controls). sizing=False is the legacy equal-weight top-bucket basket,
    which is a different, fully-invested portfolio -- useful as a signal-strength
    reference, misleading as a performance claim."""
    store = PITStore(read_only=True)
    cm = CostModel(spread_bps=spread, slippage_bps=slippage, borrow_bps_annual=borrow)
    res = run_backtest(store, backtestable_providers(), start, end, freq=freq,
                       n_buckets=n_buckets, cost_model=cm, sizing=sizing)
    store.close()
    return res.periods, res.periods_per_year, res.positions


@st.cache_data(show_spinner="Computing per-provider ICs…")
def cached_ic(start: date, end: date, freq: str):
    s = PITStore(read_only=True)
    icdf = provider_ic(s, backtestable_providers(), start, end, freq=freq)
    s.close()
    return icdf


def rebalance_records() -> list[dict]:
    """All rebalance audit records (including *-failed.json), newest first."""
    recs = []
    if REBALANCE_DIR.exists():
        for p in sorted(REBALANCE_DIR.glob("*.json"), reverse=True):
            try:
                rec = json.loads(p.read_text())
                rec["_file"] = p.name
                recs.append(rec)
            except Exception:  # noqa: BLE001  malformed record never breaks a page
                continue
    return recs


def crypto_rebalance_records() -> list[dict]:
    """Crypto trend-book audit records (outputs/rebalances_crypto/*.json), newest first."""
    recs = []
    if CRYPTO_REBALANCE_DIR.exists():
        for p in sorted(CRYPTO_REBALANCE_DIR.glob("*.json"), reverse=True):
            try:
                rec = json.loads(p.read_text())
                rec["_file"] = p.name
                recs.append(rec)
            except Exception:  # noqa: BLE001
                continue
    return recs


def corpus_status() -> dict:
    p = OUTPUTS_DIR / "corpus_status.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}
