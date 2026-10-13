"""The 2026-07-21 defect: sentiment scoring produced nothing for 8 days and every layer
reported it as benign. These tests pin the loud-failure behavior so it cannot regress.

Root cause was the chat-site model refresh deleting Ollama model `qwen3-14b`; every
/api/generate returned 404, `except Exception: return None` swallowed all 10,000 of them,
and the run logged "no output produced" and returned {"scored": 0}.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from qts.altdata.corpus import CorpusStore
from qts.nlp import sentiment_factory as sf


def _corpus_with_backlog(tmp_path, n=3):
    c = CorpusStore(tmp_path / "c.duckdb")
    c.upsert_posts(pd.DataFrame([{
        "post_id": f"p{i}", "platform": "reddit", "subreddit": "stocks",
        "author_hash": "x", "title": f"NVDA post {i}", "body": "", "score": 1,
        "num_comments": 0, "created_at": pd.Timestamp("2026-07-09"),
        "fetched_at": pd.Timestamp("2026-07-09 12:00:00"), "symbols": "NVDA"}
        for i in range(n)]))
    return c


NOW = datetime(2026, 7, 21, 9, 15, tzinfo=timezone.utc)


def test_missing_model_fails_loudly_instead_of_skipping(tmp_path, monkeypatch, capsys):
    """The exact 8-day bug: model absent on the GPU worker. Must be a `failed` result, not a
    benign skip, and must never ship a 10k batch to a box that cannot score it."""
    _corpus_with_backlog(tmp_path, n=2).close()
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: True)
    monkeypatch.setattr(sf, "_remote_model_present", lambda: False)

    def _must_not_run(*a, **k):
        raise AssertionError("shipped a batch despite the model being missing")
    monkeypatch.setattr(sf, "_run_remote", _must_not_run)

    out = sf.run(now=lambda: NOW, corpus_path=tmp_path / "c.duckdb")
    assert out["failed"] == "model_missing"
    assert sf.MODEL_NAME in capsys.readouterr().out


def test_zero_scored_from_full_batch_is_a_failure(tmp_path, monkeypatch):
    """A job that completes but scores 0 of N is an infra failure, not `scored: 0`."""
    _corpus_with_backlog(tmp_path, n=3).close()
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: True)
    monkeypatch.setattr(sf, "_remote_model_present", lambda: True)
    monkeypatch.setattr(sf, "_run_remote", lambda recs, deadline: [])

    out = sf.run(now=lambda: NOW, corpus_path=tmp_path / "c.duckdb")
    assert out["failed"] == "no_scores"


def test_partial_results_still_import(tmp_path, monkeypatch):
    """A deadline abort yields fewer rows than requested — that is success, not failure.
    Guards against the zero-of-N check over-firing."""
    _corpus_with_backlog(tmp_path, n=3).close()
    monkeypatch.setattr(sf, "_wake_and_wait", lambda d, **k: True)
    monkeypatch.setattr(sf, "_remote_model_present", lambda: True)
    monkeypatch.setattr(sf, "_run_remote",
                        lambda recs, deadline: [{"text_id": recs[0]["text_id"], "value": 0.4}])

    out = sf.run(now=lambda: NOW, corpus_path=tmp_path / "c.duckdb")
    assert out["scored"] == 1


def test_model_name_is_a_pullable_registry_name():
    """`qwen3-14b` was a local alias that vanished in the chat-site refresh. The canonical
    registry name is always re-pullable, which is what stops this recurring."""
    assert sf.MODEL_NAME == "qwen3:14b"


def test_gpu_job_preflight_exits_nonzero_when_model_absent(monkeypatch):
    """The remote job must refuse to grind through 10k no-ops against a missing model."""
    from qts.nlp import sentiment_gpu_job as job

    monkeypatch.setattr(job, "_tags", lambda: ["qwen3-30b-a3b:latest", "nomic-embed-text"])
    with pytest.raises(SystemExit) as e:
        job.preflight()
    assert e.value.code != 0


def test_gpu_job_preflight_accepts_tagged_variant(monkeypatch):
    """Ollama reports `qwen3:14b` verbatim, but tolerate a `:latest`-style suffix match."""
    from qts.nlp import sentiment_gpu_job as job

    monkeypatch.setattr(job, "_tags", lambda: ["qwen3:14b", "nomic-embed-text"])
    job.preflight()  # must not raise
