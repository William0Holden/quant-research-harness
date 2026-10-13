"""The unified Overview figure: alignment, markers, rationale, graceful degradation."""
from __future__ import annotations

import pandas as pd
import pytest

from qts.dashboard import hero


def _equity(dates, values):
    return pd.DataFrame({"ts": pd.to_datetime(dates), "equity": values})


def _bench(symbol, dates, closes):
    return pd.DataFrame({"ts": pd.to_datetime(dates), "symbol": symbol, "close": closes})


def _rec(as_of, **kw):
    rec = {"as_of": as_of, "n_orders": 10, "n_order_errors": 0,
           "trade_policy": {"turnover": 0.3, "gross": 0.58},
           "diagnostics": {"regime_gross_multiplier": 1.0,
                           "regime_state": {"risk_off": False, "vix": 16.0,
                                            "hy_oas": 2.7, "degraded": False,
                                            "fetch_errors": []}},
           "provider_weights": {"momentum_12_1": 0.23, "earnings_yield": 0.21},
           "orders": [{"symbol": "AAA", "from_w": 0.0, "to_w": 0.01},
                      {"symbol": "BBB", "from_w": 0.01, "to_w": 0.0}]}
    rec.update(kw)
    return rec


# --------------------------------------------------------------------------- #
# rebase: the benchmark must not get a longer window than the strategy
# --------------------------------------------------------------------------- #
def test_rebase_anchors_at_the_accounts_first_date():
    dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
    b = _bench("SPY", dates, [100.0, 110.0, 121.0])
    out = hero.rebase(b, "SPY", pd.Series(pd.to_datetime(dates)), 1000.0)
    assert list(out["value"]) == pytest.approx([1000.0, 1100.0, 1210.0])


def test_rebase_reindexes_onto_the_account_grid():
    """A benchmark bar on a day the account curve lacks must not add a point."""
    b = _bench("SPY", ["2026-07-01", "2026-07-02", "2026-07-03"], [100.0, 110.0, 121.0])
    grid = pd.Series(pd.to_datetime(["2026-07-01", "2026-07-03"]))
    out = hero.rebase(b, "SPY", grid, 1000.0)
    assert len(out) == 2
    assert list(out["value"]) == pytest.approx([1000.0, 1210.0])


def test_rebase_forward_fills_a_missing_benchmark_bar():
    """Account has a session the benchmark doesn't (halt, bad bar): carry, don't drop."""
    b = _bench("SPY", ["2026-07-01", "2026-07-03"], [100.0, 120.0])
    grid = pd.Series(pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]))
    out = hero.rebase(b, "SPY", grid, 100.0)
    assert list(out["value"]) == pytest.approx([100.0, 100.0, 120.0])


def test_rebase_of_a_missing_symbol_is_empty_not_an_error():
    b = _bench("SPY", ["2026-07-01"], [100.0])
    assert hero.rebase(b, "QQQ", pd.Series(pd.to_datetime(["2026-07-01"])), 1.0).empty


# --------------------------------------------------------------------------- #
# markers land where the trade took effect
# --------------------------------------------------------------------------- #
def test_marker_lands_on_the_fill_session_not_the_decision_day():
    """Friday's decision fills Monday; the marker belongs on Monday."""
    eq = _equity(["2026-07-17", "2026-07-20"], [100.0, 101.0])   # Fri, Mon
    assert hero._fill_ts("2026-07-17", eq) == pd.Timestamp("2026-07-17")
    assert hero._fill_ts("2026-07-18", eq) == pd.Timestamp("2026-07-20")


def test_marker_for_a_rebalance_after_the_last_session_pins_to_the_end():
    eq = _equity(["2026-07-17", "2026-07-20"], [100.0, 101.0])
    assert hero._fill_ts("2026-08-01", eq) == pd.Timestamp("2026-07-20")


def test_collapse_folds_a_failed_attempt_into_its_recovery():
    """2026-07-03 wrote both a failure and a recovery; plotted raw they hide each other."""
    failed = {"as_of": "2026-07-03", "error": "HON is not fractionable"}
    ok = _rec("2026-07-03")
    out = hero.collapse_by_date([failed, ok, _rec("2026-07-10")])
    assert [r["as_of"] for r in out] == ["2026-07-03", "2026-07-10"]
    assert "error" not in out[0]                       # the recovery is the outcome
    assert "fractionable" in out[0]["_retried_after"]  # the failure is kept as context
    assert "recovered after a failed attempt" in hero.rebalance_hover(out[0], None)


def test_collapse_keeps_a_date_that_only_ever_failed():
    out = hero.collapse_by_date([{"as_of": "2026-07-03", "error": "boom"}])
    assert len(out) == 1 and "error" in out[0]


# --------------------------------------------------------------------------- #
# the "why" in the hover
# --------------------------------------------------------------------------- #
def test_hover_carries_orders_turnover_gross_signals_and_trades():
    h = hero.rebalance_hover(_rec("2026-07-24"), pd.Timestamp("2026-07-27"))
    assert "Rebalance 2026-07-24" in h
    assert "filled 2026-07-27" in h
    assert "10 orders" in h and "turnover 30%" in h and "gross 58%" in h
    assert "risk-on" in h and "VIX 16.0" in h
    assert "12-1 momentum 23%" in h            # pretty label, not "momentum 12 1"
    assert "Biggest buys: AAA +1.00%" in h
    assert "Biggest sells: BBB -1.00%" in h


def test_hover_falls_back_to_sizer_gross_on_records_predating_trade_policy():
    rec = _rec("2026-07-01")
    del rec["trade_policy"]
    rec["diagnostics"]["gross"] = 0.66
    h = hero.rebalance_hover(rec, None)
    assert "gross 66%" in h
    assert "turnover" not in h                 # absent, not an em-dash


def test_hover_names_only_decision_series_as_a_blind_spot():
    """term_spread feeds no tripwire; calling it a blind spot would be misleading."""
    rec = _rec("2026-07-24")
    rec["diagnostics"]["regime_state"] = {
        "degraded": True, "fetch_errors": ["hy_oas", "term_spread", "vix"],
        "stale_series": [], "gross_multiplier": 0.6}
    rec["diagnostics"]["regime_gross_multiplier"] = 0.6
    h = hero.rebalance_hover(rec, None)
    assert "DEGRADED" in h and "hy_oas, vix unreadable" in h
    assert "term_spread" not in h
    assert "gross capped ×0.6" in h


def test_hover_calls_out_a_degraded_run_that_never_engaged_the_brake():
    """The 2026-07-24 shape: unreadable regime but full gross, because it predates the fix."""
    rec = _rec("2026-07-24")
    rec["diagnostics"]["regime_state"] = {"fetch_errors": ["hy_oas", "vix"],
                                          "gross_multiplier": 1.0}
    h = hero.rebalance_hover(rec, None)
    assert "brake did not engage" in h


def test_hover_for_a_failed_run():
    h = hero.rebalance_hover({"as_of": "2026-07-03", "error": "APIError boom"}, None)
    assert "FAILED" in h and "APIError boom" in h


def test_marker_colours_encode_state():
    assert hero._marker_color(_rec("2026-07-10")) == hero._MARKER_OK
    assert hero._marker_color({"as_of": "x", "error": "b"}) == hero._MARKER_FAIL
    assert hero._marker_color(_rec("x", kill_switch=True)) == hero._MARKER_FAIL
    assert hero._marker_color(_rec("x", n_order_errors=3)) == hero._MARKER_FAIL
    degraded = _rec("x")
    degraded["diagnostics"]["regime_state"]["degraded"] = True
    assert hero._marker_color(degraded) == hero._MARKER_WARN


# --------------------------------------------------------------------------- #
# assembly + degradation
# --------------------------------------------------------------------------- #
def _full_inputs():
    dates = pd.date_range("2026-07-01", periods=6, freq="D")
    eq = _equity(dates, [100.0, 101.0, 100.5, 102.0, 101.5, 103.0])
    bench = pd.concat([_bench(s, dates, [10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
                       for s in ("SPY", "RSP", "SGOV")], ignore_index=True)
    exposure = pd.DataFrame({"as_of": [dates[0], dates[3]], "gross": [0.6, 0.55],
                             "target_gross": [0.6, 0.6], "regime_multiplier": [1.0, 1.0],
                             "degraded": [False, False]})
    sent = pd.DataFrame({"dt": dates, "news": [0.1] * 6, "reddit": [0.0] * 6,
                         "universe_news": [0.05] * 6})
    return eq, bench, exposure, sent, [_rec(str(dates[0].date())),
                                       _rec(str(dates[3].date()))]


def test_build_hero_has_three_rows_and_every_trace():
    eq, bench, exposure, sent, recs = _full_inputs()
    fig = hero.build_hero(eq, bench, exposure, sent, recs)
    names = [t.name for t in fig.data]
    assert names[:2] == ["S&P 500 (SPY)", "S&P 500 equal-weight (RSP)"]
    for expected in ["QTS book", "Rebalance", "Invested", "Book — news",
                     "S&P 500 — news", "Book — Reddit", "T-bills (SGOV)"]:
        assert expected in names
    assert len([a for a in fig.layout.annotations]) == 3     # three subplot titles


def test_book_line_ranks_first_in_the_legend_but_draws_on_top():
    eq, bench, exposure, sent, recs = _full_inputs()
    fig = hero.build_hero(eq, bench, exposure, sent, recs)
    ranked = sorted(fig.data, key=lambda t: t.legendrank or 999)
    assert ranked[0].name == "QTS book" and ranked[1].name == "Rebalance"
    drawn = [t.name for t in fig.data]
    assert drawn.index("QTS book") > drawn.index("S&P 500 (SPY)")


def test_exposure_step_reaches_the_right_edge_of_the_figure():
    """A step that stops at the last Friday reads as 'we went flat'."""
    eq, bench, exposure, sent, recs = _full_inputs()
    sent = sent.copy()
    sent.loc[len(sent)] = {"dt": pd.Timestamp("2026-07-09"), "news": 0.1,
                           "reddit": 0.0, "universe_news": 0.05}
    fig = hero.build_hero(eq, bench, exposure, sent, recs)
    inv = next(t for t in fig.data if t.name == "Invested")
    assert pd.Timestamp(inv.x[-1]) == pd.Timestamp("2026-07-09")
    assert inv.y[-1] == pytest.approx(0.55)          # carried forward, not zeroed


def test_no_broker_data_still_renders_the_sentiment_row():
    _, _, _, sent, recs = _full_inputs()
    fig = hero.build_hero(pd.DataFrame(columns=["ts", "equity"]),
                          pd.DataFrame(columns=["ts", "symbol", "close"]),
                          pd.DataFrame(columns=["as_of", "gross"]), sent, recs)
    assert fig is not None
    assert [t.name for t in fig.data] == ["Book — news", "S&P 500 — news", "Book — Reddit"]
    assert len(fig.layout.annotations) == 1


def test_no_sentiment_drops_the_row_without_breaking_the_rest():
    eq, bench, exposure, _, recs = _full_inputs()
    fig = hero.build_hero(eq, bench, exposure, pd.DataFrame(), recs)
    assert len(fig.layout.annotations) == 2
    assert "QTS book" in [t.name for t in fig.data]


def test_no_benchmarks_still_draws_the_equity_curve():
    eq, _, exposure, sent, recs = _full_inputs()
    fig = hero.build_hero(eq, pd.DataFrame(columns=["ts", "symbol", "close"]),
                          exposure, sent, recs)
    names = [t.name for t in fig.data]
    assert "QTS book" in names and "S&P 500 (SPY)" not in names


def test_nothing_to_draw_returns_none():
    assert hero.build_hero(pd.DataFrame(columns=["ts", "equity"]),
                           pd.DataFrame(columns=["ts", "symbol", "close"]),
                           pd.DataFrame(), pd.DataFrame(), []) is None
