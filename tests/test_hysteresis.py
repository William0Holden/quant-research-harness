"""Membership hysteresis: enter from the top bucket, leave only below hold_bucket."""
from datetime import date

import pandas as pd

from qts.risk.sizing import build_long_only, hysteresis_members


def _ranked():
    # score desc; 5 = top bucket
    return pd.DataFrame({"score": [3, 2, 1, 0, -1, -2],
                         "bucket": [5, 5, 4, 4, 3, float("nan")]},
                        index=["A", "B", "C", "D", "E", "F"])


def test_held_name_in_hold_band_is_retained():
    out = hysteresis_members(_ranked(), ["A", "B"], {"C": 0.01, "E": 0.01}, hold_bucket=4)
    assert out == ["A", "B", "C"]          # C (bucket 4) stays; E (bucket 3) is sold


def test_unheld_names_below_top_are_not_admitted():
    out = hysteresis_members(_ranked(), ["A", "B"], {}, hold_bucket=4)
    assert out == ["A", "B"]                # D is bucket 4 but not held -> no entry


def test_disabled_is_identity():
    for hb in (None, 0):
        assert hysteresis_members(_ranked(), ["A", "B"], {"C": 0.01}, hb) == ["A", "B"]


def test_nan_bucket_and_zero_weight_never_retain():
    out = hysteresis_members(_ranked(), ["A"], {"F": 0.01, "C": 0.0}, hold_bucket=1)
    assert out == ["A"]


class _Store:
    def annualized_vols(self, symbols, as_of, lookback):
        return pd.Series(0.2, index=symbols)


def test_build_long_only_threads_prev_w_through_cfg():
    w_off = build_long_only(_ranked(), _Store(), date(2025, 1, 1), {}, prev_w={"C": 0.01})
    w_on = build_long_only(_ranked(), _Store(), date(2025, 1, 1), {"hold_bucket": 4},
                           prev_w={"C": 0.01})
    assert set(w_off) == {"A", "B"} and set(w_on) == {"A", "B", "C"}
    assert all(v > 0 for v in w_on.values())
