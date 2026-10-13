from datetime import date

import numpy as np
import pandas as pd

from qts.research.ic import blended_ic_weights

AS_OF = date(2026, 7, 10)


def _ic_df(n=60, good=0.05, bad=-0.03):
    idx = pd.date_range("2025-05-01", periods=n, freq="W-FRI")
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "good": good + rng.normal(0, 0.01, n),
        "bad": bad + rng.normal(0, 0.01, n),
        "flat": rng.normal(0, 0.001, n),
    }, index=idx)


def test_good_signal_outweighs_bad_and_bad_never_negative():
    w, diag = blended_ic_weights(_ic_df(), AS_OF, ["good", "bad", "flat"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["good"] > w["flat"] > 0
    assert w["bad"] >= 0                       # floored, never inverted
    assert w["good"] > w["bad"]
    # shrinkage: even a dominant signal can't exceed ~ic_share + equal share
    assert w["good"] < 0.5 * 1.0 + 0.5 * (1 / 3) + 1e-9


def test_short_long_blend_reacts_to_regime_change():
    df = _ic_df()
    # 'bad' turns strongly positive for the last 13 weeks (short-term wake-up)
    df.iloc[-13:, df.columns.get_loc("bad")] = 0.08
    w_new, _ = blended_ic_weights(df, AS_OF, ["good", "bad", "flat"])
    w_old, _ = blended_ic_weights(_ic_df(), AS_OF, ["good", "bad", "flat"])
    assert w_new["bad"] > w_old["bad"]         # short window pulls weight up fast


def test_insufficient_history_falls_back():
    w, diag = blended_ic_weights(_ic_df(n=3), AS_OF, ["good", "bad", "flat"])
    assert w is None and diag["reason"] == "insufficient_history"


def test_leak_guard_ignores_rows_on_or_after_as_of():
    df = _ic_df()
    future = pd.Timestamp(AS_OF)
    df.loc[future, :] = [9.9, 9.9, 9.9]        # poisoned same-day row
    w, diag = blended_ic_weights(df, AS_OF, ["good", "bad", "flat"])
    assert diag["n_rows_used"] == 60           # poisoned row excluded


def test_rebalance_passes_ic_weights(tmp_path, monkeypatch):
    """run_rebalance must rank with blended weights and audit them.

    blended_ic_weights itself is unit-tested above; here it is stubbed (the real
    live provider names don't appear in the toy ic_df) — this test checks the
    WIRING: load_ic_history -> blended_ic_weights -> rank_universe(weights=...)
    -> audit record."""
    import qts.exec.rebalance as rb

    captured = {}
    monkeypatch.setattr(rb, "load_ic_history", lambda: _ic_df())
    monkeypatch.setattr(
        rb, "blended_ic_weights",
        lambda ic, as_of, names, **kw: (
            {n: 1.0 / len(names) for n in names}, {"n_rows_used": 60}))

    def spy_rank(store, providers, symbols, as_of, weights=None, **kw):
        captured["weights"] = weights
        return pd.DataFrame({"score": [], "bucket": []})

    monkeypatch.setattr(rb, "rank_universe", spy_rank)
    monkeypatch.setattr(rb, "build_target_portfolio",
                        lambda *a, **k: ({}, {"gross": 0.0}))

    class FakeStore:
        def universe_as_of(self, as_of):
            return pd.DataFrame({"symbol": ["AAA"], "name": ["A"]})

    from qts.exec.broker import DryRunBroker
    rec = rb.run_rebalance(FakeStore(), AS_OF, DryRunBroker(), out_dir=tmp_path)
    assert captured["weights"] is not None
    assert rec["provider_weights"] is not None
    assert abs(sum(rec["provider_weights"].values()) - 1.0) < 1e-4
