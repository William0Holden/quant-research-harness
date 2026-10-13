import pandas as pd
from qts.nlp import factory as NF
from qts.pit.store import PITStore


def _filing(sym, acc, fdate, form="10-K", text="body " * 1000):
    return {"symbol": sym, "cik": "1", "form": form,
            "period_end": pd.Timestamp(fdate).date(),
            "filing_date": pd.Timestamp(fdate).date(), "accession": acc,
            "section": "full", "text": text, "knowledge_time": pd.Timestamp(fdate)}


def test_export_incremental_with_context(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    store.write_filings(pd.DataFrame([
        _filing("AAA", "a1", "2023-02-01"), _filing("AAA", "a2", "2024-02-01"),
        _filing("AAA", "q1", "2024-05-01", form="10-Q"),
    ]))
    # a1 already scored -> only a2 (new, with a1 as context) and q1 (new, no predecessor)
    store.write_nlp_features(pd.DataFrame([{
        "symbol": "AAA", "concept": "filing_sentiment",
        "filing_date": pd.Timestamp("2023-02-01").date(), "period_end": None,
        "value": 0.1, "knowledge_time": pd.Timestamp("2023-02-01")}]))
    recs = NF.export_sections(store)
    by_acc = {r["accession"]: r for r in recs}
    assert set(by_acc) == {"a1", "a2", "q1"}
    assert by_acc["a1"]["context_only"] is True
    assert by_acc["a2"]["context_only"] is False and by_acc["a2"]["form"] == "10-K"
    assert by_acc["q1"]["context_only"] is False and by_acc["q1"]["form"] == "10-Q"
    store.close()


def test_export_nothing_new(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    store.write_filings(pd.DataFrame([_filing("AAA", "a1", "2023-02-01")]))
    store.write_nlp_features(pd.DataFrame([{
        "symbol": "AAA", "concept": "filing_sentiment",
        "filing_date": pd.Timestamp("2023-02-01").date(), "period_end": None,
        "value": 0.1, "knowledge_time": pd.Timestamp("2023-02-01")}]))
    assert NF.export_sections(store) == []
    store.close()


def test_max_docs_cap(tmp_path):
    store = PITStore(tmp_path / "t.duckdb")
    store.write_filings(pd.DataFrame(
        [_filing("AAA", f"a{i}", f"20{10+i}-02-01") for i in range(6)]))
    recs = NF.export_sections(store, max_docs=2)
    new = [r for r in recs if not r["context_only"]]
    assert len(new) == 2
    # newest-first: the two most recent filings are the scored ones
    assert {r["accession"] for r in new} == {"a5", "a4"}
    store.close()


def test_export_dedupes_same_day_filings(tmp_path):
    # Regression for the same-day PK collision (nlp_features PK is (symbol, concept,
    # filing_date)): a 10-Q and a same-day 8-K press release must collapse to a single
    # non-context record, keeping the more substantive form.
    store = PITStore(tmp_path / "t.duckdb")
    store.write_filings(pd.DataFrame([
        _filing("BBB", "q1", "2024-05-01", form="10-Q"),
        _filing("BBB", "p1", "2024-05-01", form="8-K"),
    ]))
    recs = NF.export_sections(store)
    non_ctx = [r for r in recs if not r["context_only"]]
    assert len(non_ctx) == 1
    assert non_ctx[0]["form"] == "10-Q"
    assert non_ctx[0]["filing_date"].startswith("2024-05-01")
    store.close()


def test_import_features_dedupes_same_day_rows(tmp_path):
    # Belt-and-braces: even if the GPU job somehow emits two rows sharing a
    # (symbol, concept, filing_date), import_features must not raise a
    # ConstraintException and should write exactly one row.
    store = PITStore(tmp_path / "t.duckdb")
    feats = [
        {"symbol": "BBB", "concept": "filing_sentiment", "filing_date": "2024-05-01",
         "period_end": None, "value": 0.2},
        {"symbol": "BBB", "concept": "filing_sentiment", "filing_date": "2024-05-01",
         "period_end": None, "value": -0.5},
    ]
    n = NF.import_features(store, feats)
    assert n == 1
    df = store.con.execute("SELECT * FROM nlp_features").df()
    assert len(df) == 1
    store.close()
