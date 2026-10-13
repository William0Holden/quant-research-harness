"""Crypto linker: cashtag + curated alias ONLY. The bare-English-word-ticker trap is the
whole reason this exists, so most of these tests are things that must NOT link."""
from qts.crypto.sentiment.linker import link_symbols, link_text
from qts.crypto.universe import BTC, ETH


def test_cashtags_link_any_case():
    assert link_text("loading up on $BTC here") == {BTC}
    assert link_text("$eth and $btc both ripping") == {BTC, ETH}
    assert link_text("$XBT is just bitcoin") == {BTC}


def test_curated_aliases_link_word_bounded():
    assert link_text("Bitcoin broke 65k") == {BTC}
    assert link_text("ethereum gas is high") == {ETH}
    assert link_text("i bought some ether today") == {ETH}


def test_bare_english_word_tickers_NEVER_link():
    """The core failure the equity linker would cause. None of these may link a coin."""
    for txt in ("please LINK me the article",
                "the DOT at the end of the sentence",
                "a UNI degree", "swing that BAT", "reach for the SKY",
                "the HYPE is real", "TRUMP said something", "ARB is arbitrary"):
        assert link_text(txt) == set(), f"false link on: {txt!r}"


def test_substrings_do_not_false_match():
    # "ethernet"/"bitcoiners" must not match via word boundaries... "bitcoiners" contains
    # "bitcoin" as a prefix but \b makes "bitcoin" match inside "bitcoiners"? No: \b is at a
    # word char/non-word-char boundary, and 'bitcoin' followed by 'ers' has no boundary, so
    # \bbitcoin\b does NOT match 'bitcoiners'. Pin that.
    assert link_text("ethernet cable") == set()          # 'ether' not word-bounded here
    assert link_text("bitcoiners are numerous") == set()  # 'bitcoin' not word-bounded in 'bitcoiners'
    # but a standalone 'bitcoin' (even used as a verb) IS a real, bounded mention:
    assert link_text("just gonna bitcoin it") == {BTC}


def test_link_symbols_storage_format():
    assert link_symbols("$BTC and ethereum") == f"{BTC},{ETH}" or \
           link_symbols("$BTC and ethereum") == f"{ETH},{BTC}"  # sorted, comma-joined
    assert link_symbols("nothing here") == ""
    # sorted deterministically
    assert link_symbols("ethereum bitcoin") == ",".join(sorted([BTC, ETH]))
