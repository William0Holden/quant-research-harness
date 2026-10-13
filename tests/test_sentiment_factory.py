from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from qts.altdata.corpus import CorpusStore
from qts.nlp import sentiment_factory as sf


@pytest.fixture(autouse=True)
def _no_real_gpu_probe(monkeypatch):
    """Keep tests hermetic: never SSH to a real GPU worker for the model-presence check."""
    monkeypatch.setattr(sf, "_remote_model_present", lambda: True)


def _corpus_with_backlog(tmp_path, n=3):
    c = CorpusStore(tmp_path / "c.duckdb")
    c.upsert_posts(pd.DataFrame([{
        "post_id": f"p{i}", "platform": "reddit", "subreddit": "stocks",
        "author_hash": "x", "title": f"NVDA post {i}", "body": "", "score": 1,
        "num_comments": 0, "created_at": pd.Timestamp("2026-07-09"),
        "fetched_at": pd.Timestamp("2026-07-09 12:00:00"), "symbols": "NVDA"}
        for i in range(n)]))
    return c


def test_run_no_longer_skips_by_wall_clock(tmp_path, monkeypatch):
    """The whole point of the 2026-07-19 change: a run late in the day must NOT skip. With the
    old absolute 11:50 window this returned {'skipped':'past_deadline'}; now it scores whenever
    the GPU worker is available, any time of day."""
    _corpus_with_backlog(tmp_path, n=2).close()
    late = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)   # 8pm UTC — was past the old window
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: True)
    monkeypatch.setattr(sf, "_run_remote",
                        lambda recs, deadline: [{"text_id": r["text_id"], "value": 0.1}
                                                for r in recs])
    out = sf.run(now=lambda: late, corpus_path=tmp_path / "c.duckdb")
    assert out["scored"] == 2                                  # scored, not skipped


def test_run_skips_cleanly_when_gpu_unavailable(tmp_path, monkeypatch):
    _corpus_with_backlog(tmp_path).close()
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: False)
    out = sf.run(now=lambda: datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc),
                 corpus_path=tmp_path / "c.duckdb")
    assert out == {"skipped": "gpu_unavailable"}


def test_run_scores_and_imports(tmp_path, monkeypatch):
    _corpus_with_backlog(tmp_path, n=2).close()
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: True)
    monkeypatch.setattr(sf, "_run_remote",
                        lambda recs, deadline: [{"text_id": r["text_id"], "value": 0.25}
                                                for r in recs])
    out = sf.run(now=lambda: datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
                 corpus_path=tmp_path / "c.duckdb")
    assert out["scored"] == 2
    with CorpusStore(tmp_path / "c.duckdb", read_only=True) as c2:
        assert c2.unscored_texts().empty
        assert c2.con.execute("SELECT value FROM text_sentiment LIMIT 1").fetchone()[0] == 0.25


def test_run_noop_when_no_backlog(tmp_path):
    CorpusStore(tmp_path / "c.duckdb").close()
    out = sf.run(now=lambda: datetime(2026, 7, 10, 9, 15, tzinfo=timezone.utc),
                 corpus_path=tmp_path / "c.duckdb")
    assert out == {"skipped": "no_backlog"}


def test_wake_and_wait_gives_up_within_budget_when_unreachable(monkeypatch):
    """The politeness guarantee: an unreachable box (busy box, ssh closed) must make
    _wake_and_wait return False in bounded time, not spin. No real sleeping or WoL."""
    monkeypatch.setattr(sf, "_ssh_ok", lambda: False)          # never reachable
    monkeypatch.setattr(sf.subprocess, "run", lambda *a, **k: None)  # WoL is a noop
    slept = {"total": 0}
    monkeypatch.setattr(sf.time, "sleep", lambda s: slept.__setitem__("total", slept["total"] + s))
    ok = sf._wake_and_wait(lambda: False, wake_budget_s=60)
    assert ok is False
    assert slept["total"] <= 60 + sf.WAKE_POLL_S               # bounded, did not spin forever


def test_per_run_budget_bounds_the_deadline(tmp_path, monkeypatch):
    """The elapsed-time deadline closure: once max_run_minutes passes, past_deadline is True so
    _run_remote stops. Simulate an advancing clock."""
    _corpus_with_backlog(tmp_path, n=1).close()
    t = {"now": datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)}
    def clock():
        t["now"] += timedelta(minutes=60)                     # jump > budget each call
        return t["now"]
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: True)
    seen = {}
    def fake_remote(recs, deadline):
        seen["past_deadline_after_advancing"] = deadline()    # by now >45 min elapsed
        return []
    monkeypatch.setattr(sf, "_run_remote", fake_remote)
    sf.run(now=clock, corpus_path=tmp_path / "c.duckdb", max_run_minutes=45)
    assert seen["past_deadline_after_advancing"] is True
