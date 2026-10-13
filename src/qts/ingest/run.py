"""Ingestion orchestrator: universe -> prices -> SEC fundamentals/events."""
from __future__ import annotations

from datetime import date

from ..pit.store import PITStore
from ..pit.sp500 import build_sp500_universe
from .prices import ingest_prices
from .sec import ingest_sec


def run_full_ingest(db_path=None, do_sec: bool = True, incremental: bool = False) -> dict:
    store = PITStore() if db_path is None else PITStore(db_path)
    try:
        nu = build_sp500_universe(store)
        members = store.universe_as_of(date.today())
        syms = list(members["symbol"])
        print(f"universe: {nu} rows, {len(syms)} active members")

        print("ingesting prices (Yahoo)...")
        pr = ingest_prices(store, syms, incremental=incremental)
        ok_pr = sum(1 for v in pr.values() if v > 0)
        print(f"  prices ok for {ok_pr}/{len(syms)} symbols")

        sec = {}
        if do_sec:
            print("ingesting SEC fundamentals + events (EDGAR)...")
            sec = ingest_sec(store, members[members["cik"].notna()])
            ok_f = sum(1 for v in sec.values() if v.get("funda", -1) > 0)
            print(f"  fundamentals ok for {ok_f}/{len(sec)} symbols")

        print("final counts:", store.table_counts())
        return {"universe": nu, "prices": pr, "sec": sec, "counts": store.table_counts()}
    finally:
        store.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="qts data ingest")
    ap.add_argument("--incremental", action="store_true",
                    help="resume prices from last ingested bar - 7d")
    ap.add_argument("--no-sec", action="store_true",
                    help="skip SEC fundamentals/events (prices only)")
    a = ap.parse_args()
    run_full_ingest(do_sec=not a.no_sec, incremental=a.incremental)
