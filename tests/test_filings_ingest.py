"""tests/test_filings_ingest.py"""
import pandas as pd
import pytest

from qts.ingest import filings as F
from qts.pit.store import PITStore


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


SUBS = {"filings": {"recent": {
    "accessionNumber": ["0001-24-1", "0001-24-2", "0001-24-3", "0001-24-4"],
    "form": ["10-K", "10-Q", "8-K", "10-Q"],
    "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
    "filingDate": ["2024-02-01", "2024-05-01", "2024-06-01", "2024-08-01"],
    "reportDate": ["2023-12-31", "2024-03-31", "", "2024-06-30"],
    "items": ["", "", "2.02,9.01", ""],
}}}


def test_list_filings_multi_form(monkeypatch):
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: FakeResp(SUBS))
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    out = F.list_filings("0000000001", forms=("10-K", "10-Q"), limit_per_form=1)
    assert [o["form"] for o in out] == ["10-K", "10-Q"]
    assert out[0]["items"] == ""


def test_list_filings_8k_items(monkeypatch):
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: FakeResp(SUBS))
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    out = F.list_filings("0000000001", forms=("8-K",), limit_per_form=5)
    assert len(out) == 1 and out[0]["items"] == "2.02,9.01"


def test_ingest_skips_existing_accessions(tmp_path, monkeypatch):
    store = PITStore(tmp_path / "t.duckdb")
    store.write_filings(pd.DataFrame([{
        "symbol": "AAA", "cik": "0000000001", "form": "10-K",
        "period_end": pd.Timestamp("2023-12-31").date(),
        "filing_date": pd.Timestamp("2024-02-01").date(),
        "accession": "0001-24-1", "section": "full", "text": "old",
        "knowledge_time": pd.Timestamp("2024-02-01"),
    }]))
    monkeypatch.setattr(F, "list_filings",
                        lambda cik, forms=("10-K",), limit_per_form=6: [
                            {"form": "10-K", "accession": "0001-24-1", "doc": "a.htm",
                             "filing_date": "2024-02-01", "period_end": "2023-12-31", "items": ""}])
    fetched = []
    monkeypatch.setattr(F, "fetch_document_text",
                        lambda cik, acc, doc: fetched.append(acc) or "x" * 6000)
    members = pd.DataFrame([{"symbol": "AAA", "cik": "0000000001"}])
    out = F.ingest_filings(store, members)
    assert fetched == [] and out["AAA"] == 0
    store.close()


TENQ = (
    "part i item 2. management's discussion and analysis of financial condition "
    + "growth was strong this quarter. " * 60
    + " item 3. quantitative and qualitative disclosures about market risk. "
    + "part ii item 1a. risk factors " + "there have been no material changes. " * 40
    + " item 6. exhibits."
)

def test_extract_sections_10q():
    out = F.extract_sections(TENQ, form="10-Q")
    assert "mda" in out and "growth was strong" in out["mda"]
    assert "risk_factors" in out and "no material changes" in out["risk_factors"]
    assert "full" in out


def test_find_ex99_doc(monkeypatch):
    idx = {"directory": {"item": [
        {"name": "form8k.htm"}, {"name": "ex99-1.htm"}, {"name": "ex99_1_long_name.htm"},
        {"name": "img99.jpg"}]}}
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: FakeResp(idx))
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    assert F.find_ex99_doc("0000000001", "0001-24-3") == "ex99-1.htm"


def test_ingest_press_releases(tmp_path, monkeypatch):
    store = PITStore(tmp_path / "t.duckdb")
    monkeypatch.setattr(F, "list_filings",
                        lambda cik, forms, limit_per_form: [
                            {"form": "8-K", "accession": "0001-24-3", "doc": "c.htm",
                             "filing_date": "2024-06-01", "period_end": None,
                             "items": "2.02,9.01"},
                            {"form": "8-K", "accession": "0001-24-9", "doc": "e.htm",
                             "filing_date": "2024-07-01", "period_end": None,
                             "items": "8.01"}])
    monkeypatch.setattr(F, "find_ex99_doc", lambda cik, acc: "ex99-1.htm")
    monkeypatch.setattr(F, "fetch_document_text",
                        lambda cik, acc, doc: "record quarterly revenue " * 100)
    members = pd.DataFrame([{"symbol": "AAA", "cik": "0000000001"}])
    out = F.ingest_press_releases(store, members)
    rows = store.con.execute(
        "SELECT form, section, accession FROM filings").df()
    assert out["AAA"] == 1 and len(rows) == 1          # only the 2.02 filing
    assert rows.iloc[0]["section"] == "press_release"
    store.close()
