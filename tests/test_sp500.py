from datetime import date

import pytest

from qts.pit.sp500 import load_fallback, parse_sp500_html
from qts.pit import sp500 as sp500_mod
from qts.pit.store import PITStore

FIXTURE = """
<table id="constituents"><thead>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr></thead><tbody>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
<tr><td>MMM</td><td>3M</td><td>Industrials</td></tr>
</tbody></table>
"""


def test_parse_sp500_html_normalizes_symbols():
    rows = parse_sp500_html(FIXTURE)
    syms = {r["symbol"] for r in rows}
    assert syms == {"AAPL", "BRK-B", "MMM"}          # dot -> dash (Yahoo form)
    assert {r["name"] for r in rows} == {"Apple Inc.", "Berkshire Hathaway", "3M"}


def test_load_fallback_is_big_enough():
    rows = load_fallback()
    assert len(rows) > 480
    assert all("." not in r["symbol"] for r in rows)


def test_build_universe_preserves_delisted_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(sp500_mod, "fetch_sp500",
                        lambda: parse_sp500_html(FIXTURE))
    monkeypatch.setattr(sp500_mod, "_cik_map", lambda: {"AAPL": "0000320193"})
    monkeypatch.setattr(sp500_mod, "_save_fallback", lambda rows: None)
    with PITStore(tmp_path / "t.duckdb") as store:
        n = sp500_mod.build_sp500_universe(store)
        assert n >= 4                                 # 3 constituents + FRC
        u = store.universe_as_of(date(2022, 1, 1))
        assert "FRC" in set(u["symbol"])              # delisted seed intact
        u2 = store.universe_as_of(date(2026, 1, 1))
        assert "FRC" not in set(u2["symbol"])
        aapl = u[u["symbol"] == "AAPL"].iloc[0]
        assert aapl["cik"] == "0000320193"
