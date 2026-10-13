"""tests/test_live_gate.py"""
from datetime import date

import numpy as np
import pandas as pd

from qts import config as C
from qts.exec.broker import DryRunBroker
from qts.exec.rebalance import run_rebalance
from qts.pit.store import PITStore
from qts.types import Backtestability


def test_default_gate_covers_new_providers():
    from qts.features.base import default_providers
    names = {p.name for p in default_providers()}
    expected = {"event_officer_change", "event_non_reliance", "eightk_freq_anomaly",
                "news_volume_anomaly", "reddit_mention_anomaly"}
    assert expected <= names
    assert expected <= C.DEFAULT_LIVE_GATED


def test_env_override(monkeypatch):
    monkeypatch.setenv("QTS_LIVE_GATED", "news_volume_anomaly, foo")
    assert C.live_gated() == frozenset({"news_volume_anomaly", "foo"})
    monkeypatch.setenv("QTS_LIVE_GATED", "")
    assert C.live_gated() == frozenset()
    monkeypatch.delenv("QTS_LIVE_GATED")
    assert C.live_gated() == C.DEFAULT_LIVE_GATED


def _seed_store(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    syms = [f"SYM{i}" for i in range(8)]
    store.write_universe(pd.DataFrame({
        "symbol": syms, "cik": None, "name": syms,
        "active_from": date(2020, 1, 1), "active_to": None, "delisted": False,
    }))
    dts = pd.bdate_range("2024-01-01", "2025-06-30")
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(syms):
        px = 100 * np.cumprod(1 + 0.0002 * i + 0.01 * rng.standard_normal(len(dts)))
        rows.extend({
            "symbol": s, "dt": d.date(), "close": p, "adj_close": p, "volume": 1000,
            "knowledge_time": d + pd.Timedelta(hours=23),
        } for d, p in zip(dts, px))
    store.write_prices(pd.DataFrame(rows))
    return store, syms


def test_rebalance_excludes_gated_and_logs_them(tmp_path, monkeypatch):
    monkeypatch.delenv("QTS_LIVE_GATED", raising=False)
    store, syms = _seed_store(tmp_path)
    out_dir = tmp_path / "rebalances"
    try:
        rec = run_rebalance(store, date(2025, 6, 15), DryRunBroker(),
                            regime_on=False, out_dir=out_dir)
    finally:
        store.close()

    gated = C.live_gated()
    assert set(rec["live_providers"]).isdisjoint(gated)
    assert "gated_signals" in rec
    assert set(rec["gated_signals"]) <= gated
    # Event providers emit 0.0 on any store, so the audit record has entries for them.
    assert len(rec["gated_signals"]) > 0


def test_dashboard_backtest_uses_backtestable_providers():
    """Task-9 integration fix: dashboard Backtest tab must not pass forward-only
    providers into run_backtest (leak guard #6 raises ValueError on those)."""
    from qts.features.base import default_providers
    forward_only = [p for p in default_providers()
                    if p.spec.backtestability != Backtestability.BACKTESTABLE]
    assert forward_only, "expected at least one forward-only provider registered"

    common_src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "qts" / "dashboard" / "common.py"
    ).read_text()
    assert "run_backtest(store, backtestable_providers()" in common_src
    assert "run_backtest(store, default_providers()" not in common_src


def test_dashboard_portfolio_tab_uses_live_providers():
    """Review finding: the Portfolio tab's caption claims 'what the Alpaca adapter
    would trade' but was ranking with default_providers() (all providers, including
    IC-gated ones) rather than the same live-gated composite qts.exec.rebalance
    actually trades with. Assert the tab calls the live_providers() helper."""
    view_src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "qts" / "dashboard" / "views" / "portfolio.py"
    ).read_text()
    assert "live_providers()" in view_src
    assert "rank_universe(store, live_providers()" in view_src
