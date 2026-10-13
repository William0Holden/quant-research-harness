import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

import qts.exec.rebalance as rebalance_mod
from qts.exec.broker import DryRunBroker
from qts.exec.rebalance import _equity_history, pick_broker, run_rebalance
from qts.pit.store import PITStore


@pytest.fixture
def seeded(tmp_path):
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
    yield store, syms
    store.close()


def test_run_rebalance_dry_run_audit(seeded, tmp_path):
    store, syms = seeded
    out_dir = tmp_path / "rebalances"
    rec = run_rebalance(store, date(2025, 6, 15), DryRunBroker(),
                        regime_on=False, out_dir=out_dir)
    assert rec["broker"] == "DryRunBroker"
    assert rec["submitted"] is False
    assert 0 < sum(rec["weights"].values()) <= 1.0 + 1e-9
    assert all(o["dry_run"] for o in rec["orders"])
    on_disk = json.loads((out_dir / "2025-06-15.json").read_text())
    assert on_disk["as_of"] == "2025-06-15"
    assert on_disk["n_orders"] == len(rec["orders"])
    assert rec["equity_source"] == "fallback"


def test_kill_switch_flattens(seeded, tmp_path):
    store, _ = seeded
    out_dir = tmp_path / "rebalances"
    # Fabricate REAL broker history alone that already breaches the 25% drawdown —
    # the trip must come from this history, not from the current run's fallback
    # equity (a DryRunBroker run has no real account, so its equity must never
    # dilute/pollute a real broker's drawdown series).
    for d, eq in [("2025-05-02", 100_000.0), ("2025-05-09", 70_000.0)]:
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / f"{d}.json").write_text(
            json.dumps({"as_of": d, "equity": eq, "equity_source": "broker"}))
    rec = run_rebalance(store, date(2025, 6, 15), DryRunBroker(),
                        equity_fallback=100_000.0, regime_on=False, out_dir=out_dir)
    assert rec["equity_source"] == "fallback"
    assert rec["kill_switch"] is True
    assert rec["weights"] == {}
    # flattening: every order is a sell/no-op, never a buy
    assert all(o["side"] != "buy" for o in rec["orders"])


def test_fallback_equity_does_not_trip_kill_switch(seeded, tmp_path):
    store, _ = seeded
    out_dir = tmp_path / "rebalances"
    # Fabricate REAL broker history with a 14% drawdown (140k -> 120k), which is
    # below the 25% threshold. When fallback equity is supplied, it should be
    # excluded from drawdown calculation (per _equity_history), so kill_switch
    # remains False.
    for d, eq in [("2025-05-02", 140_000.0), ("2025-05-09", 120_000.0)]:
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / f"{d}.json").write_text(
            json.dumps({"as_of": d, "equity": eq, "equity_source": "broker"}))
    rec = run_rebalance(store, date(2025, 6, 15), DryRunBroker(),
                        equity_fallback=100_000.0, regime_on=False, out_dir=out_dir)
    assert rec["equity_source"] == "fallback"
    assert rec["kill_switch"] is False
    assert rec["weights"] != {}


def test_pick_broker_defaults_dry(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert isinstance(pick_broker(), DryRunBroker)


def test_main_writes_failure_record_and_exits_nonzero(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("db locked")

    monkeypatch.setattr(rebalance_mod, "PITStore", boom)
    monkeypatch.setattr(rebalance_mod, "DEFAULT_OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["rebalance", "--as-of", "2026-07-03"])

    with pytest.raises(SystemExit) as exc:
        rebalance_mod.main()
    assert exc.value.code == 1

    rec = json.loads((tmp_path / "2026-07-03-failed.json").read_text())
    assert "error" in rec
    assert rec["as_of"] == "2026-07-03"
    assert rec["submitted"] is False


def test_equity_history_skips_fallback_records(tmp_path):
    out_dir = tmp_path / "rebalances"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"as_of": "2025-05-02", "equity": 100000.0, "equity_source": "broker"},
        {"as_of": "2025-05-09", "equity": 50.0, "equity_source": "fallback"},
        {"as_of": "2025-05-16", "equity": 99000.0},
    ]
    for rec in records:
        (out_dir / f"{rec['as_of']}.json").write_text(json.dumps(rec))
    assert _equity_history(out_dir) == [100000.0, 99000.0]


class _DroppingBroker(DryRunBroker):
    """A venue that cannot trade SYM0: the runner must drop it before ranking."""

    def filter_tradable(self, symbols):
        return [s for s in symbols if s != "SYM0"], {"SYM0": "no active tradable asset on venue"}


class _BrokenAssetListBroker(DryRunBroker):
    def filter_tradable(self, symbols):
        raise RuntimeError("asset list unreachable")


def test_untradable_names_never_receive_weight(seeded, tmp_path):
    store, syms = seeded
    rec = run_rebalance(store, date(2025, 6, 15), _DroppingBroker(),
                        regime_on=False, out_dir=tmp_path / "r")
    assert "SYM0" not in rec["target_weights"] and "SYM0" not in rec["weights"]
    assert rec["untradable"] == {"SYM0": "no active tradable asset on venue"}
    assert sum(rec["weights"].values()) > 0          # the rest of the book still trades


def test_asset_list_outage_fails_open_and_is_recorded(seeded, tmp_path):
    store, syms = seeded
    rec = run_rebalance(store, date(2025, 6, 15), _BrokenAssetListBroker(),
                        regime_on=False, out_dir=tmp_path / "r")
    assert sum(rec["weights"].values()) > 0
    assert "asset list unreachable" in rec["untradable"]["_error"]
