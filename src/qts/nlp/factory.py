"""NLP factory orchestration — RUNS ON the main host. Exports filing sections, ships them to
GPU worker, runs the Ollama batch job there, imports the resulting PIT-stamped features.

This is the concrete wiring of the design's "NLP = batch job submitted to GPU worker" seam.
It drives GPU worker directly over SSH (GPU worker is WoL-woken); the same gpu_job.py is also
submittable via an external job runner for a cron-driven production path.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from ..config import OUTPUTS_DIR, ensure_dirs
from ..pit.store import PITStore

# GPU worker transport -- configure via env (see .env.example).
GPU_HOST = os.environ.get("QTS_GPU_HOST", "gpu-worker.local")
SSH_PORT = os.environ.get("QTS_GPU_SSH_PORT", "22")
SSH_USER = os.environ.get("QTS_GPU_SSH_USER", "qts")
SSH_KEY = os.path.expanduser(os.environ.get("QTS_GPU_SSH_KEY", "~/.ssh/id_ed25519"))
REMOTE_DIR = os.environ.get("QTS_GPU_REMOTE_DIR", "nlp-job")
SECTION_PRIORITY = ["risk_factors", "mda", "full"]
# Same-day filing priority for the (symbol, filing_date) dedupe below: the report
# dominates a same-day press release.
_FORM_PRIORITY = {"10-K": 0, "10-Q": 1, "8-K": 2}

# ServerAlive*: a GPU worker reboot/OS-switch mid-session leaves an ESTAB socket that never
# RSTs; without keepalives the launch ssh hung 10h+ (2026-08-22) and wedged the flock.
_SSH = ["ssh", "-i", SSH_KEY, "-p", SSH_PORT, "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4", f"{SSH_USER}@{GPU_HOST}"]
_SCP = ["scp", "-i", SSH_KEY, "-P", SSH_PORT, "-o", "BatchMode=yes"]


def _pick_section(rows: dict) -> dict:
    """Prefer risk_factors > mda when substantive (>3000 chars), else fall back to full —
    avoids thin TOC/summary stubs that give noisy sentiment."""
    for s in ("risk_factors", "mda"):
        if s in rows and len(rows[s].get("text") or "") > 3000:
            return rows[s]
    return rows.get("full") or next(iter(rows.values()))


def _scored_keys(store: PITStore) -> set[tuple[str, str]]:
    df = store.con.execute(
        "SELECT DISTINCT symbol, filing_date FROM nlp_features "
        "WHERE concept = 'filing_sentiment'").df()
    return {(r.symbol, str(r.filing_date)) for r in df.itertuples()}


def export_sections(store: PITStore, max_docs: int = 400) -> list[dict]:
    """One chosen section per filing, NEW filings only (not yet scored), newest-first
    capped at max_docs, plus each new filing's same-form predecessor as a
    context_only record (embedded for the Lazy-Prices diff, never re-scored).

    nlp_features' PK is (symbol, concept, filing_date) -- a company that files, say,
    an 8-K press release the same day as its 10-K/10-Q would otherwise produce two
    output rows sharing that PK and blow up the whole GPU import batch. So before any
    of the scored/new/context logic runs, filings are deduped to one per (symbol,
    filing_date), preferring the more substantive form (10-K > 10-Q > 8-K > anything
    else) via _FORM_PRIORITY. A press release filed the same day as a 10-K/10-Q is
    therefore intentionally skipped -- the report dominates it."""
    fx = store.filings_export()
    if fx.empty:
        return []
    groups: dict[tuple, dict] = {}
    for _, r in fx.iterrows():
        groups.setdefault((r["symbol"], r["accession"]), {})[r["section"]] = r.to_dict()
    chosen = [_pick_section(rows) for rows in groups.values()]

    best: dict[tuple, tuple[int, dict]] = {}
    for r in chosen:
        key = (r["symbol"], str(r["filing_date"]))
        prio = _FORM_PRIORITY.get(r.get("form") or "10-K", len(_FORM_PRIORITY))
        cur = best.get(key)
        if cur is None or prio < cur[0]:
            best[key] = (prio, r)
    chosen = [r for _, r in best.values()]

    scored = _scored_keys(store)

    def _rec(r: dict, context_only: bool) -> dict:
        return {"symbol": r["symbol"], "accession": r["accession"],
                "form": r.get("form") or "10-K",
                "filing_date": str(r["filing_date"]),
                "period_end": str(r["period_end"]) if pd.notna(r["period_end"]) else None,
                "text": r["text"] or "", "context_only": context_only}

    new = [r for r in chosen if (r["symbol"], str(r["filing_date"])) not in scored]
    new.sort(key=lambda r: str(r["filing_date"]), reverse=True)
    new = new[:max_docs]
    picked = {(r["symbol"], r["accession"]) for r in new}

    # same-form predecessor of each new filing -> context anchor for the diff
    by_sf: dict[tuple, list[dict]] = {}
    for r in chosen:
        by_sf.setdefault((r["symbol"], r.get("form") or "10-K"), []).append(r)
    ctx = []
    for lst in by_sf.values():
        lst.sort(key=lambda r: str(r["filing_date"]))
        for i, r in enumerate(lst):
            if (r["symbol"], r["accession"]) in picked and i > 0:
                prev = lst[i - 1]
                if (prev["symbol"], prev["accession"]) not in picked:
                    ctx.append(prev)
                    picked.add((prev["symbol"], prev["accession"]))
    return [_rec(r, True) for r in ctx] + [_rec(r, False) for r in new]


def run_on_gpu(records: list[dict], poll_s: int = 20, timeout_s: int | None = None) -> list[dict]:
    timeout_s = timeout_s or max(3600, 30 * len(records))
    ensure_dirs()
    inp = OUTPUTS_DIR / "nlp_input.json"
    outp = OUTPUTS_DIR / "nlp_output.json"
    inp.write_text(json.dumps(records))
    job = str(Path(__file__).with_name("gpu_job.py"))

    subprocess.run([*_SSH, f"mkdir -p {REMOTE_DIR}"], check=True)
    subprocess.run([*_SCP, job, f"{SSH_USER}@{GPU_HOST}:{REMOTE_DIR}/gpu_job.py"], check=True)
    subprocess.run([*_SCP, str(inp), f"{SSH_USER}@{GPU_HOST}:{REMOTE_DIR}/input.json"], check=True)

    # launch detached; write a done-marker on completion
    remote_cmd = (f"cd {REMOTE_DIR} && nohup bash -c "
                  f"'python3 gpu_job.py input.json output.json > job.log 2>&1; touch DONE' "
                  f">/dev/null 2>&1 & echo started")
    subprocess.run([*_SSH, remote_cmd], check=True)

    waited = 0
    while waited < timeout_s:
        time.sleep(poll_s)
        waited += poll_s
        done = subprocess.run([*_SSH, f"test -f {REMOTE_DIR}/DONE && echo yes || echo no"],
                              capture_output=True, text=True).stdout.strip()
        tail = subprocess.run([*_SSH, f"tail -1 {REMOTE_DIR}/job.log 2>/dev/null"],
                              capture_output=True, text=True).stdout.strip()
        print(f"  [{waited}s] done={done} | {tail}")
        if done == "yes":
            break
    else:
        raise TimeoutError("gpu nlp job did not finish in time")

    subprocess.run([*_SCP, f"{SSH_USER}@{GPU_HOST}:{REMOTE_DIR}/output.json", str(outp)], check=True)
    subprocess.run([*_SSH, f"rm -f {REMOTE_DIR}/DONE"], check=False)
    return json.loads(outp.read_text())


def import_features(store: PITStore, feats: list[dict]) -> int:
    if not feats:
        return 0
    df = pd.DataFrame(feats)
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce").dt.date
    df["knowledge_time"] = pd.to_datetime(df["filing_date"])
    # Belt-and-braces: nlp_features PK is (symbol, concept, filing_date); even with the
    # export-side dedupe above, guard the import path too (gpu_job.py output is an
    # independent hop and could in principle still double up).
    df = df.drop_duplicates(subset=["symbol", "concept", "filing_date"], keep="first")
    df = df[["symbol", "concept", "filing_date", "period_end", "value", "knowledge_time"]]
    return store.write_nlp_features(df)


def run_nlp_factory(db_path=None) -> dict:
    store = PITStore() if db_path is None else PITStore(db_path)
    try:
        recs = export_sections(store)
        print(f"exporting {len(recs)} filings to GPU worker...")
        if not recs:
            return {"filings": 0, "features": 0}
        feats = run_on_gpu(recs)
        n = import_features(store, feats)
        print(f"imported {n} nlp feature rows; counts: {store.table_counts()}")
        return {"filings": len(recs), "features": n}
    finally:
        store.close()


if __name__ == "__main__":
    run_nlp_factory()
