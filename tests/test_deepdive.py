import json
from datetime import date
from types import SimpleNamespace

import pandas as pd

from qts.altdata.corpus import CorpusStore
from qts.altdata.deepdive import BING_TICKER, find_spikes, run_deepdive


def _news_row(h, sym, dt):
    return {"url_hash": h, "url": f"https://x/{h}", "source": "s", "title": h,
            "summary": "", "body": None, "claimed_published_at": None,
            "fetched_at": pd.Timestamp(dt), "symbols": sym,
            "title_hash": h, "extract_ok": False}


def test_find_spikes_flags_burst_not_steady(tmp_path):
    today = date(2026, 7, 10)
    with CorpusStore(tmp_path / "c.duckdb") as c:
        rows = []
        # STEADY: 2 mentions every day incl. today -> no spike
        for d in range(14):
            dt = pd.Timestamp(today) - pd.Timedelta(days=d)
            rows += [_news_row(f"s{d}a", "STDY", dt), _news_row(f"s{d}b", "STDY", dt)]
        # BRST: silent for 13 days, 8 mentions today -> spike
        rows += [_news_row(f"b{i}", "BRST", pd.Timestamp(today)) for i in range(8)]
        c.upsert_news(pd.DataFrame(rows))
        spikes = find_spikes(c, today=today)
        assert spikes == ["BRST"]


def test_run_deepdive_fetches_top_bodies_and_caps(tmp_path, monkeypatch):
    # isolate from the real outputs/corpus_status.json once-per-day ledger
    from qts.altdata import deepdive
    monkeypatch.setattr(deepdive, "_done_today", lambda: set())
    monkeypatch.setattr(deepdive, "_mark_done", lambda syms: None)
    fetched = []

    def parse(url):
        assert "bing.com/news" in url
        es = [SimpleNamespace(link=f"https://pub.example/{i}", title=f"t{i}",
                              summary="s", published_parsed=None)
              for i in range(15)]
        return SimpleNamespace(entries=es)

    def fetch_body(url):
        fetched.append(url)
        return "full article body", True

    with CorpusStore(tmp_path / "c.duckdb") as c:
        out = run_deepdive(c, {"NVDA"}, {}, ["NVDA"], parse=parse,
                           fetch_body=fetch_body)
        assert out["new_items"] == 10 and len(fetched) == 10   # top-10 cap
        src = c.con.execute("SELECT DISTINCT source FROM news_items").fetchone()[0]
        assert src == "bing-deepdive-NVDA"
        body = c.con.execute("SELECT body FROM news_items LIMIT 1").fetchone()[0]
        assert body == "full article body"
