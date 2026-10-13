"""tests/test_news_collector.py"""
import json
import types
import pandas as pd
from qts.altdata import news as N
from qts.altdata.corpus import CorpusStore


def _entry(link, title, summary="s", published_parsed=None):
    return types.SimpleNamespace(link=link, title=title, summary=summary,
                                 published_parsed=published_parsed)


def _fake_parse_factory(mapping):
    def parse(url):
        return types.SimpleNamespace(entries=mapping.get(url, []), bozo=False)
    return parse


def test_collect_dedup_and_link(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    feed_url = N.BROAD_FEEDS[0][1]
    mapping = {feed_url: [
        _entry("https://x.test/a?utm_source=rss", "TSLA surges on record deliveries"),
        _entry("https://x.test/a", "TSLA surges on record deliveries"),   # dup after canon
        _entry("https://x.test/b", "quiet market day"),
    ]}
    with CorpusStore(tmp_path / "c.duckdb") as c:
        res = N.collect_news(
            c, symbols={"TSLA"}, alias_map={},
            parse=_fake_parse_factory(mapping),
            fetch_body=lambda url: ("body text", True),
            max_articles=10, batch=0, state_path=tmp_path / "rot.json")
        assert c.counts()["news_items"] == 2                      # canon-dedup worked
        df = c.con.execute("SELECT url, symbols FROM news_items ORDER BY url").df()
        assert "utm_source" not in df.iloc[0]["url"]
        assert dict(zip(df["url"], df["symbols"]))["https://x.test/a"] == "TSLA"
        assert res["new_items"] == 2


def test_rerun_skips_known(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    feed_url = N.BROAD_FEEDS[0][1]
    mapping = {feed_url: [_entry("https://x.test/a", "hello market")]}
    fetches = []
    with CorpusStore(tmp_path / "c.duckdb") as c:
        for _ in range(2):
            N.collect_news(c, symbols=set(), alias_map={},
                           parse=_fake_parse_factory(mapping),
                           fetch_body=lambda url: (fetches.append(url) or "b", True),
                           max_articles=10, batch=0, state_path=tmp_path / "rot.json")
        assert c.counts()["news_items"] == 1 and len(fetches) == 1


def test_broad_yahoo_finance_feed_does_not_seed_bogus_ticker(tmp_path, monkeypatch):
    # Regression for finding 1: the broad "yahoo-finance" feed must NOT seed the
    # bogus ticker "finance" via source-name prefix matching.
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    mapping = {}
    for source, url in N.BROAD_FEEDS:
        entry_link = f"https://x.test/{source}"
        mapping[url] = [_entry(entry_link, "no tickers mentioned here at all")]
    with CorpusStore(tmp_path / "c.duckdb") as c:
        N.collect_news(
            c, symbols={"TSLA"}, alias_map={},
            parse=_fake_parse_factory(mapping),
            fetch_body=lambda url: (None, False),
            max_articles=10, batch=0, state_path=tmp_path / "rot.json")
        df = c.con.execute("SELECT source, symbols FROM news_items").df()
        yahoo_finance_row = df[df["source"] == "yahoo-finance"]
        assert len(yahoo_finance_row) == 1
        assert yahoo_finance_row.iloc[0]["symbols"] == ""
        assert not (df["symbols"].str.contains("finance")).any()


def test_malformed_published_parsed_stores_null_and_does_not_raise(tmp_path, monkeypatch):
    # Regression for finding 2: a malformed published_parsed must not blow up the
    # whole collection run; the row should still land with claimed_published_at NULL.
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    feed_url = N.BROAD_FEEDS[0][1]
    mapping = {feed_url: [
        _entry("https://x.test/bad-date", "garbage date market news",
               published_parsed=("not", "a", "real", "time", "tuple", "here", "x", "y", "z"))
    ]}
    with CorpusStore(tmp_path / "c.duckdb") as c:
        res = N.collect_news(
            c, symbols=set(), alias_map={},
            parse=_fake_parse_factory(mapping),
            fetch_body=lambda url: (None, False),
            max_articles=10, batch=0, state_path=tmp_path / "rot.json")
        assert res["new_items"] == 1
        df = c.con.execute(
            "SELECT claimed_published_at FROM news_items WHERE url = "
            "'https://x.test/bad-date'").df()
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["claimed_published_at"])


def test_rotation_advances(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    state = tmp_path / "rot.json"
    syms = {f"S{i:03d}" for i in range(100)}
    seen_urls = []
    def parse(url):
        seen_urls.append(url)
        return types.SimpleNamespace(entries=[], bozo=False)
    with CorpusStore(tmp_path / "c.duckdb") as c:
        N.collect_news(c, symbols=syms, alias_map={}, parse=parse,
                       fetch_body=lambda u: (None, False), max_articles=5,
                       batch=40, state_path=state)
        off1 = json.loads(state.read_text())["offset"]
        N.collect_news(c, symbols=syms, alias_map={}, parse=parse,
                       fetch_body=lambda u: (None, False), max_articles=5,
                       batch=40, state_path=state)
        off2 = json.loads(state.read_text())["offset"]
    assert off1 == 40 and off2 == 80


def test_google_ticker_rotation_and_no_gnews_bodies(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    symbols = {"AAA", "BBB", "CCC"}
    state = tmp_path / "rot.json"
    seen_urls, body_urls = [], []

    def parse(url):
        seen_urls.append(url)
        e = _entry(f"https://news.google.com/rss/articles/x{len(seen_urls)}?oc=5",
                   f"t{len(seen_urls)}")
        return types.SimpleNamespace(entries=[e], bozo=False)

    def fetch_body(url):
        body_urls.append(url)
        return "body", True

    with CorpusStore(tmp_path / "c.duckdb") as c:
        N.collect_news(c, symbols, {}, parse=parse, fetch_body=fetch_body,
                       batch=2, state_path=state)
    # NB: two BROAD_FEEDS are also news.google.com/rss/search — filter per-ticker
    # ones by the quoted-symbol marker (%22) the broad queries don't have.
    gnews = [u for u in seen_urls
             if "news.google.com/rss/search" in u and "%22" in u]
    assert len(gnews) == 2                      # g-rotation ran with batch=2
    assert body_urls == []                      # never fetch bodies for gnews links
    st = json.loads(state.read_text())
    assert st["offset"] == 2 and st["g_offset"] == 2
