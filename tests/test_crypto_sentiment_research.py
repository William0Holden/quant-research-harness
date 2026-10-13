"""The measurement harness grades against RISK and refuses to score on thin data."""
import numpy as np
import pandas as pd
import pytest

from qts.crypto.sentiment.research import (ICResult, SignalScorecard, evaluate, rank_ic,
                                          MIN_PAIRS)


def _dates(n):
    return pd.date_range("2026-01-01", periods=n, freq="D")


def test_insufficient_history_is_reported_not_faked():
    s = pd.Series([1, 2, 3], index=_dates(3))
    t = pd.Series([0.1, 0.2, 0.3], index=_dates(3))
    r = rank_ic(s, t, "sig", "fwd_vol")
    assert not np.isfinite(r.ic)
    assert "insufficient" in r.note and r.n == 3


def test_detects_a_real_positive_risk_ic():
    n = 60
    rng = np.random.default_rng(0)
    sig = pd.Series(rng.normal(size=n), index=_dates(n))
    target = sig * 0.8 + rng.normal(scale=0.3, size=n)   # target tracks the signal
    r = rank_ic(sig, target, "attention", "fwd_vol")
    assert r.n == n and r.ic > 0.5 and r.t_stat > 2


def test_no_relationship_gives_insignificant_ic():
    n = 80
    rng = np.random.default_rng(1)
    sig = pd.Series(rng.normal(size=n), index=_dates(n))
    target = pd.Series(rng.normal(size=n), index=_dates(n))   # independent
    r = rank_ic(sig, target, "sig", "fwd_dd")
    assert abs(r.t_stat) < 2                                  # correctly finds nothing


def test_alignment_drops_unpaired_and_nan_days():
    n = 40
    sig = pd.Series(range(n), index=_dates(n), dtype=float)
    target = sig.copy()
    target.iloc[:15] = np.nan                                # first 15 days unscored
    r = rank_ic(sig, target, "s", "t")
    assert r.n == 25                                         # only paired, non-NaN days


def test_scorecard_only_earns_on_significant_t():
    sc = SignalScorecard()
    sc.add(ICResult("vol_forecast", "fwd_vol", n=60, ic=0.4, t_stat=3.1))
    sc.add(ICResult("attention_spike", "fwd_vol", n=60, ic=0.1, t_stat=0.8))
    sc.add(ICResult("equity_risk_appetite", "fwd_dd", n=10, note="insufficient history (n=10<20)"))
    assert sc.earned(min_t=2.0) == ["vol_forecast"]         # only the significant one


def test_evaluate_from_dicts_aligns_by_date():
    sig = {f"2026-01-{d:02d}": d for d in range(1, 31)}
    tgt = {f"2026-01-{d:02d}": d * 2.0 for d in range(1, 31)}
    r = evaluate(sig, tgt, "s", "fwd_vol")
    assert r.n == 30 and r.ic == pytest.approx(1.0, abs=1e-6)
