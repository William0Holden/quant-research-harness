"""The simulated book and the traded book must not be able to drift apart.

This is the bug class that already bit once: the Reddit comment phase was reachable from
`reddit_browser.main()` but NOT from the entrypoint cron actually ran, so a feature that
passed its tests never executed in production. Same shape here, higher stakes -- if the
backtest validated a turnover policy the live rebalance didn't apply (or vice versa),
every number we used to choose the parameters would describe a portfolio we don't trade.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

import qts.backtest.engine as eng
import qts.exec.rebalance as rb
import qts.risk.trade_policy as tp
from qts.exec.broker import DryRunBroker

AS_OF = date(2026, 7, 10)


def test_both_paths_bind_the_same_policy_function():
    """Not 'both have a band' -- literally the same object. An inline reimplementation
    on either side is exactly how the two books silently diverge."""
    assert rb.apply_trade_policy is tp.apply_trade_policy
    assert eng.apply_trade_policy is tp.apply_trade_policy


class _FakeStore:
    def universe_as_of(self, as_of):
        return pd.DataFrame({"symbol": ["AAA", "BBB"], "name": ["A", "B"]})


def _stub_pipeline(monkeypatch, target):
    monkeypatch.setattr(rb, "load_ic_history", lambda: pd.DataFrame())
    monkeypatch.setattr(rb, "rank_universe",
                        lambda *a, **k: pd.DataFrame({"score": [], "bucket": []}))
    monkeypatch.setattr(rb, "build_target_portfolio",
                        lambda *a, **k: (dict(target), {"gross": sum(target.values())}))
    monkeypatch.setattr(rb, "default_providers", lambda: [])


def test_live_rebalance_applies_the_policy_to_the_sized_target(tmp_path, monkeypatch):
    """A held name whose target moved less than the band must not generate an order."""
    _stub_pipeline(monkeypatch, {"AAA": 0.102, "BBB": 0.20})

    broker = DryRunBroker()
    broker._positions = {"AAA": 0.10}            # 20bp drift, inside a 50bp band

    rec = rb.run_rebalance(
        _FakeStore(), AS_OF, broker, out_dir=tmp_path,
        policy={"band_w": 0.005, "lam": 1.0, "dust_w": 0.0025},
    )

    traded = {o["symbol"] for o in rec["orders"]}
    assert "AAA" not in traded                   # suppressed by the band
    assert "BBB" in traded                       # a real new position still trades
    assert rec["weights"]["AAA"] == 0.10         # held where it was, not dropped
    assert rec["target_weights"]["AAA"] == 0.102  # cost-free ideal still audited
    assert rec["trade_policy"]["n_banded"] == 1


def test_kill_switch_bypasses_partial_adjustment(tmp_path, monkeypatch):
    """The kill switch LIQUIDATES. A partial adjustment would leave the book half-on in
    the exact drawdown the switch exists to escape -- so the policy must be bypassed."""
    _stub_pipeline(monkeypatch, {})
    monkeypatch.setattr(rb, "_equity_history", lambda d: [100_000.0, 50_000.0])  # -50% DD

    broker = DryRunBroker()
    broker._positions = {"AAA": 0.10, "BBB": 0.10}

    rec = rb.run_rebalance(
        _FakeStore(), AS_OF, broker, out_dir=tmp_path,
        equity_fallback=50_000.0,
        policy={"band_w": 0.005, "lam": 0.25, "dust_w": 0.0025},   # lam would keep 75%
    )

    assert rec["kill_switch"] is True
    assert rec["weights"] == {}                                     # fully flat
    assert rec["trade_policy"] == {"bypassed": "kill_switch"}
    assert {o["symbol"] for o in rec["orders"]} == {"AAA", "BBB"}   # both liquidated
    assert all(o["side"] == "sell" for o in rec["orders"])
