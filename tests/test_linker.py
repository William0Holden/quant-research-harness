import pandas as pd
from qts.altdata.linker import build_alias_map, link_text


MEMBERS = pd.DataFrame([
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "GD", "name": "General Dynamics Corporation"},
    {"symbol": "A", "name": "Agilent Technologies Inc."},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
])
SYMS = set(MEMBERS["symbol"])


def test_cashtag_and_bare_ticker():
    amap = build_alias_map(MEMBERS)
    assert link_text("Loaded up on $TSLA calls", SYMS, amap) == {"TSLA"}
    assert link_text("AAPL beats on earnings", SYMS, amap) == {"AAPL"}


def test_stoplist_blocks_ambiguous():
    amap = build_alias_map(MEMBERS)
    # 'A' is a real ticker but a stop word as a bare token; cashtag still works
    assert link_text("A big day for stocks", SYMS, amap) == set()
    assert link_text("bought $A today", SYMS, amap) == {"A"}


def test_multiword_name_match():
    amap = build_alias_map(MEMBERS)
    assert link_text("General Dynamics wins contract", SYMS, amap) == {"GD"}
    # single-word alias never matches by name (precision over recall)
    assert link_text("apple pie recipe", SYMS, amap) == set()


def test_suffix_stripping_is_end_anchored():
    # Regression: unanchored suffix stripping used to strip "Group" out of the
    # middle of the name, collapsing the core to "american international" and
    # falsely matching unrelated text containing those two words.
    members = pd.DataFrame([
        {"symbol": "AIG", "name": "American International Group, Inc."},
    ])
    syms = {"AIG"}
    amap = build_alias_map(members)
    assert amap == {"american international group": "AIG"}
    assert link_text(
        "The American International Toy Fair kicks off today.", syms, amap
    ) == set()
    assert link_text("American International Group posts profit", syms, amap) == {"AIG"}


def test_suffix_stripping_keeps_group_as_brand_word():
    # Regression: "Group" used to be in the strippable suffix set, which
    # collapsed "CBRE Group, Inc." to the single word "cbre" and dropped it
    # from the alias map entirely.
    members = pd.DataFrame([
        {"symbol": "CBRE", "name": "CBRE Group, Inc."},
    ])
    amap = build_alias_map(members)
    assert amap == {"cbre group": "CBRE"}


def test_ampersand_stripped_from_core():
    # Regression: "JPMorgan Chase & Co." stripped its "Co." suffix but left a
    # dangling "&" on the core (strip(" ,.") didn't include "&"), so the alias
    # never matched plain "JPMorgan Chase" mentions.
    members = pd.DataFrame([
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    ])
    syms = {"JPM"}
    amap = build_alias_map(members)
    assert amap == {"jpmorgan chase": "JPM"}
    assert link_text("JPMorgan Chase posts record profit", syms, amap) == {"JPM"}


def test_ambiguous_alias_core_removed_entirely():
    # Two different symbols producing the same core is an ambiguous alias --
    # neither should get a name-matching entry (cashtags/bare tickers still
    # cover them).
    members = pd.DataFrame([
        {"symbol": "ABI", "name": "Alpha Beta Inc."},
        {"symbol": "ABC", "name": "Alpha Beta Corp."},
    ])
    amap = build_alias_map(members)
    assert "alpha beta" not in amap
    assert amap == {}
