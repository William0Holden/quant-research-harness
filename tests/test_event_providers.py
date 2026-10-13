"""tests/test_event_providers.py"""
from datetime import date
import pandas as pd
from qts.features.events import EightKFreqAnomaly, NonReliance, OfficerChange
from qts.pit.store import PITStore


def _ev(sym, item, fdate, acc):
    return {"symbol": sym, "cik": "1", "form": "8-K", "item": item,
            "filing_date": pd.Timestamp(fdate).date(), "accession": acc,
            "extra": None, "knowledge_time": pd.Timestamp(fdate)}


def _store(tmp_path, rows):
    s = PITStore(tmp_path / "t.duckdb")
    s.write_events(pd.DataFrame(rows))
    return s


def test_officer_change_decays(tmp_path):
    s = _store(tmp_path, [_ev("AAA", "5.02", "2024-06-25", "x1"),
                          _ev("BBB", "5.02", "2024-04-10", "x2"),
                          _ev("CCC", "2.02", "2024-06-25", "x3")])
    v = OfficerChange().compute(s, ["AAA", "BBB", "CCC", "DDD"], date(2024, 6, 30))
    assert v["AAA"] > v["BBB"] > 0          # fresher event scores higher
    assert v["CCC"] == 0.0 and v["DDD"] == 0.0
    s.close()


def test_non_reliance_flag(tmp_path):
    s = _store(tmp_path, [_ev("AAA", "4.02", "2024-01-15", "y1")])
    v = NonReliance().compute(s, ["AAA", "BBB"], date(2024, 6, 30))
    assert v["AAA"] == 1.0 and v["BBB"] == 0.0
    s.close()


def test_freq_anomaly_pit(tmp_path):
    rows = [_ev("AAA", "8.01", f"2024-06-{d:02d}", f"z{d}") for d in range(1, 9)]
    rows += [_ev("AAA", "8.01", "2023-09-01", "old1")]
    s = _store(tmp_path, rows)
    hot = EightKFreqAnomaly().compute(s, ["AAA"], date(2024, 6, 30))["AAA"]
    # before the burst existed, anomaly must be lower (PIT sanity)
    cold = EightKFreqAnomaly().compute(s, ["AAA"], date(2024, 5, 1))["AAA"]
    assert hot > cold
    s.close()
