"""Crypto collection links coins correctly and stores them in corpus convention. Uses a fake
feed parser + an in-memory corpus -- no network."""
import pandas as pd
import pytest

from qts.altdata.corpus import CorpusStore
from qts.crypto.sentiment.collect import collect_news_crypto
from qts.crypto.universe import BTC, ETH


class _FakeEntry:
    def __init__(self, title, summary, link):
        self.title, self.summary, self.link = title, summary, link


class _FakeFeed:
    def __init__(self, entries):
        self.entries = entries


def _parser(entries):
    return lambda url: _FakeFeed(entries)


def test_news_links_coins_and_stores_symbols(tmp_path):
    db = tmp_path / "corpus.duckdb"
    entries = [
        _FakeEntry("Bitcoin ETF sees inflows", "big day for $BTC", "http://x/1"),
        _FakeEntry("Ethereum upgrade ships", "ethereum devs happy", "http://x/2"),
        _FakeEntry("Fed holds rates", "nothing crypto here", "http://x/3"),
        _FakeEntry("LINK to the article about DOT products", "no coins", "http://x/4"),
    ]
    with CorpusStore(db) as corpus:
        out = collect_news_crypto(corpus, parse=_parser(entries), fetch_body=lambda u: (None, False))
        assert out["new_news"] == 4
        rows = corpus.con.execute(
            "SELECT title, symbols FROM news_items ORDER BY url").df()
    bysym = dict(zip(rows["title"], rows["symbols"]))
    assert bysym["Bitcoin ETF sees inflows"] == BTC
    assert bysym["Ethereum upgrade ships"] == ETH
    assert bysym["Fed holds rates"] == ""                      # no coin -> no link
    # THE trap: 'LINK' and 'DOT' are English words here, must NOT link coins
    assert bysym["LINK to the article about DOT products"] == ""


def test_news_dedups_on_rerun(tmp_path):
    db = tmp_path / "corpus.duckdb"
    entries = [_FakeEntry("Bitcoin news", "$BTC", "http://x/1")]
    with CorpusStore(db) as corpus:
        assert collect_news_crypto(corpus, parse=_parser(entries),
                                   fetch_body=lambda u: (None, False))["new_news"] == 1
        # second pass: same URL -> zero new
        assert collect_news_crypto(corpus, parse=_parser(entries),
                                   fetch_body=lambda u: (None, False))["new_news"] == 0


def test_browser_engine_accepts_a_crypto_linkfn(tmp_path):
    """The refactor: run the shared browser engine with the CRYPTO linker injected. Crypto
    coins link; English-word tickers (LINK/DOT) do not; and the equity default path is
    untouched (covered by the equity suite)."""
    from qts.altdata.reddit_browser import collect_posts_browser
    from qts.crypto.sentiment.linker import link_text as crypto_link

    raw_posts = [
        {"id": "p1", "title": "Bitcoin to the moon", "selftext": "$BTC breakout",
         "subreddit": "Bitcoin", "author": "a", "score": 10, "num_comments": 3,
         "created_utc": 1_700_000_000, "stickied": False},
        {"id": "p2", "title": "LINK and DOT discussion", "selftext": "no majors here",
         "subreddit": "CryptoCurrency", "author": "b", "score": 5, "num_comments": 1,
         "created_utc": 1_700_000_100, "stickied": False},
    ]
    with CorpusStore(tmp_path / "c.duckdb") as corpus:
        out = collect_posts_browser(corpus, set(), {}, fetch=lambda: raw_posts,
                                    link_fn=crypto_link)
        assert out["new_posts"] == 2
        rows = dict(corpus.con.execute(
            "SELECT post_id, symbols FROM social_posts").df().values)
    assert rows["p1"] == BTC                       # bitcoin/$BTC linked
    assert rows["p2"] == ""                         # LINK/DOT are English words -> no link


def test_equity_default_linkfn_still_used_when_none(tmp_path):
    """Guard: with link_fn=None the engine must fall back to the equity linker exactly, so
    the production equity path is unchanged by the refactor."""
    from qts.altdata.reddit_browser import collect_posts_browser
    raw = [{"id": "e1", "title": "Apple earnings", "selftext": "AAPL up",
            "subreddit": "stocks", "author": "a", "score": 1, "num_comments": 0,
            "created_utc": 1_700_000_000, "stickied": False}]
    with CorpusStore(tmp_path / "c.duckdb") as corpus:
        # equity linker matches bare ticker AAPL from the provided symbol set
        out = collect_posts_browser(corpus, {"AAPL"}, {}, fetch=lambda: raw)
        assert out["new_posts"] == 1
        sym = corpus.con.execute("SELECT symbols FROM social_posts").fetchone()[0]
    assert sym == "AAPL"
