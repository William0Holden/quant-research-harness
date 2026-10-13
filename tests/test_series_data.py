"""Exposure and sentiment series that explain the equity curve."""
from __future__ import annotations

import pandas as pd
import pytest

from qts.dashboard import series_data as SD


def _rec(as_of, gross=0.58, target=0.62, mult=1.0, regime=None, **kw):
    rec = {
        "as_of": as_of,
        "trade_policy": {"gross": gross, "target_gross": target},
        "diagnostics": {"gross": target, "regime_gross_multiplier": mult,
                        "regime_state": regime if regime is not None
                        else {"degraded": False, "fetch_errors": []}},
    }
    rec.update(kw)
    return rec


# --------------------------------------------------------------------------- #
# exposure
# --------------------------------------------------------------------------- #
def test_exposure_uses_the_held_gross_not_the_target():
    df = SD.exposure_series([_rec("2026-07-24", gross=0.58, target=0.62)])
    assert df.loc[0, "gross"] == pytest.approx(0.58)
    assert df.loc[0, "target_gross"] == pytest.approx(0.62)


def test_exposure_falls_back_to_sizer_gross_before_trade_policy_existed():
    rec = _rec("2026-07-01")
    del rec["trade_policy"]
    rec["diagnostics"]["gross"] = 0.66
    df = SD.exposure_series([rec])
    assert df.loc[0, "gross"] == pytest.approx(0.66)


def test_exposure_skips_failed_runs_and_sorts():
    df = SD.exposure_series([_rec("2026-07-24"), {"as_of": "2026-07-10", "error": "boom"},
                             _rec("2026-07-17")])
    assert list(df["as_of"].dt.strftime("%Y-%m-%d")) == ["2026-07-17", "2026-07-24"]


def test_exposure_flags_a_degraded_regime_read():
    explicit = SD.exposure_series([_rec("2026-07-24", regime={"degraded": True})])
    assert bool(explicit.loc[0, "degraded"]) is True
    # Records predating the `degraded` key: a non-empty fetch_errors stands in for it.
    legacy = SD.exposure_series([_rec("2026-07-24", regime={"fetch_errors": ["vix"]})])
    assert bool(legacy.loc[0, "degraded"]) is True
    errored = SD.exposure_series([_rec("2026-07-01", regime={"error": "fetch failed"})])
    assert bool(errored.loc[0, "degraded"]) is True
    assert bool(SD.exposure_series([_rec("2026-07-17")]).loc[0, "degraded"]) is False


def test_exposure_with_no_usable_records_is_empty_not_an_error():
    assert SD.exposure_series([]).empty
    assert SD.exposure_series([{"as_of": "2026-07-01", "error": "x"}]).empty


# --------------------------------------------------------------------------- #
# which book was held when
# --------------------------------------------------------------------------- #
def test_held_weights_use_the_prior_rebalance_not_the_same_day():
    """Friday's decision fills Monday, so Friday itself still runs the old book."""
    recs = [_rec("2026-07-10", weights={"A": 0.5}), _rec("2026-07-17", weights={"B": 0.5})]
    dates = pd.to_datetime(pd.Series(["2026-07-13", "2026-07-17", "2026-07-20"]))
    got = SD._held_weights_by_date(recs, dates)
    assert got[pd.Timestamp("2026-07-13")] == {"A": 0.5}
    assert got[pd.Timestamp("2026-07-17")] == {"A": 0.5}      # not B yet
    assert got[pd.Timestamp("2026-07-20")] == {"B": 0.5}


def test_dates_before_the_first_rebalance_have_no_book():
    recs = [_rec("2026-07-10", weights={"A": 1.0})]
    got = SD._held_weights_by_date(recs, pd.to_datetime(pd.Series(["2026-07-01"])))
    assert got == {}


def test_failed_and_weightless_records_are_ignored():
    recs = [{"as_of": "2026-07-10", "error": "boom", "weights": {"Z": 1.0}},
            _rec("2026-07-11", weights={}),
            _rec("2026-07-12", weights={"A": 1.0})]
    got = SD._held_weights_by_date(recs, pd.to_datetime(pd.Series(["2026-07-15"])))
    assert got[pd.Timestamp("2026-07-15")] == {"A": 1.0}


# --------------------------------------------------------------------------- #
# sentiment weighting
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel(monkeypatch):
    """Two symbols, opposite news moods, very different article counts."""
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2026-07-20"] * 2),
        "symbol": ["A", "B"],
        "news": [1.0, -1.0],
        "news_n": [90.0, 10.0],
        "reddit": [0.5, float("nan")],
        "reddit_n": [4.0, 0.0],
    })
    monkeypatch.setattr(SD, "sentiment_panel", lambda start: df)
    return df


def test_book_sentiment_weights_by_position_size(panel):
    """The book is mostly B, so book sentiment must lean negative even though the
    universe (article-count weighted) leans strongly positive."""
    recs = [_rec("2026-07-10", weights={"A": 0.1, "B": 0.9})]
    out = SD.book_sentiment(recs, "2026-07-01")
    assert out.loc[0, "news"] == pytest.approx(0.1 * 1.0 + 0.9 * -1.0)
    assert out.loc[0, "universe_news"] == pytest.approx((90 * 1.0 + 10 * -1.0) / 100)


def test_universe_is_item_count_weighted_not_a_mean_of_means(panel):
    out = SD.book_sentiment([], "2026-07-01")
    assert out.loc[0, "universe_news"] == pytest.approx(0.8)   # not 0.0


def test_symbols_outside_the_book_are_excluded(panel):
    out = SD.book_sentiment([_rec("2026-07-10", weights={"A": 1.0})], "2026-07-01")
    assert out.loc[0, "news"] == pytest.approx(1.0)


def test_missing_sentiment_does_not_count_as_neutral(panel):
    """B has no Reddit score; averaging it in as 0 would bury A's signal."""
    recs = [_rec("2026-07-10", weights={"A": 0.5, "B": 0.5})]
    out = SD.book_sentiment(recs, "2026-07-01")
    assert out.loc[0, "reddit"] == pytest.approx(0.5)          # A's value, not 0.25


def test_book_sentiment_without_a_book_still_returns_the_universe(panel):
    out = SD.book_sentiment([], "2026-07-01")
    assert list(out.columns) == ["dt", "news", "reddit", "universe_news"]
    assert pd.isna(out.loc[0, "news"])
    assert pd.notna(out.loc[0, "universe_news"])


def test_empty_panel_yields_an_empty_frame(monkeypatch):
    monkeypatch.setattr(SD, "sentiment_panel", lambda start: pd.DataFrame(
        columns=["dt", "symbol", "news", "news_n", "reddit", "reddit_n"]))
    out = SD.book_sentiment([_rec("2026-07-10", weights={"A": 1.0})], "2026-07-01")
    assert out.empty and list(out.columns) == ["dt", "news", "reddit", "universe_news"]
