"""tests/test_reddit_collector.py"""
import types
from qts.altdata import news as N
from qts.altdata import reddit as R
from qts.altdata.corpus import CorpusStore


def _post(pid, title, body="", score=10, sub="stocks"):
    return types.SimpleNamespace(
        id=pid, title=title, selftext=body, score=score, num_comments=3,
        created_utc=1751400000.0, author=types.SimpleNamespace(name="u1"),
        subreddit=types.SimpleNamespace(display_name=sub))


class FakeSub:
    def __init__(self, posts):
        self._posts = posts
    def new(self, limit=100):
        return iter(self._posts[:limit])


class FakeReddit:
    def __init__(self, by_sub):
        self._by = by_sub
    def subreddit(self, name):
        return FakeSub(self._by.get(name, []))


def test_collect_posts_links_and_dedups(tmp_path, monkeypatch):
    # reddit.collect_posts calls news._update_status at the end, which writes
    # OUTPUTS_DIR/corpus_status.json -- patch the news module's global (reddit.py
    # imports the function, not the constant, so this is the only place that matters)
    # so the test never touches the live outputs/corpus_status.json.
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    rd = FakeReddit({"stocks": [_post("p1", "loaded $TSLA"), _post("p2", "macro talk")],
                     "wallstreetbets": [_post("p1", "loaded $TSLA")]})   # cross-listed dup id
    with CorpusStore(tmp_path / "c.duckdb") as c:
        out = R.collect_posts(rd, c, symbols={"TSLA"}, alias_map={},
                              subreddits=("stocks", "wallstreetbets"))
        assert c.counts()["social_posts"] == 2 and out["new_posts"] == 2
        df = c.con.execute(
            "SELECT post_id, symbols, author_hash FROM social_posts ORDER BY post_id").df()
        assert dict(zip(df["post_id"], df["symbols"]))["p1"] == "TSLA"
        assert df.iloc[0]["author_hash"] != "u1"          # author is hashed, never raw
        assert (tmp_path / "corpus_status.json").exists()  # status went to tmp_path...


def test_make_reddit_none_without_creds(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OUTPUTS_DIR", tmp_path)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    assert R.make_reddit() is None
