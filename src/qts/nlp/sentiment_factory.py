"""Social/news sentiment scorer — RUNS ON the main host; GPU work on the GPU worker.

CADENCE (user, 2026-07-19): the old 4-8am-only window is relaxed. The scorer now runs SEVERAL
times a day (cron slots) and each run works for a bounded budget, so the backlog drains over
successive slots and the scorer catches GPU worker whenever it is briefly available -- instead of
one daily shot that misses for a week when the box is unavailable.

The user is NEVER fought for the machine: every run WoL-wakes GPU worker but gives up fast
(MAX_WAKE_WAIT_S) if it is not reachable -- which is exactly what happens when the user is on
it (other OS -> ssh closed). A reachable-but-in-use Linux box is the only case where we might
share the GPU, and the job is short (one BATCH_CAP batch) and abortable.

Per-run budget: MAX_RUN_MINUTES caps the GPU phase; partial output IS imported (the gpu job
writes incrementally); the remainder re-queues via the idempotent unscored_texts export.

Corpus lock is held ONLY around the export and import phases (fcntl on the same
outputs/.qts-corpus.lock the shell collectors flock) — never across the GPU phase, so hourly
collectors keep running. A separate flock in deploy/sentiment-scorer.sh prevents two slots
from overlapping."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..altdata.corpus import CorpusStore
from ..config import OUTPUTS_DIR, ensure_dirs
from .factory import _SCP, _SSH, GPU_HOST, SSH_USER

BATCH_CAP = 10_000   # measured 2026-07-10: ~5.8 texts/s on qwen3-14b -> 10k ~= 30 min
# Per-run budget: each cron slot works at most this long, then aborts+imports partial output.
# Several slots per day drain a large backlog. Sized above the ~30-min BATCH_CAP runtime.
MAX_RUN_MINUTES = 45
# Give up fast if the GPU worker is not reachable after a WoL wake -- a box that is busy or booted into another OS
# never becomes reachable, so this is what keeps the scorer from fighting the user for it.
MAX_WAKE_WAIT_S = 180
WAKE_POLL_S = 20
JOB_POLL_S = 30
REMOTE_DIR = os.environ.get("QTS_GPU_SENTIMENT_REMOTE_DIR", "sentiment-job")
# Optional wake command (e.g. a Wake-on-LAN wrapper) run before polling SSH.
GPU_WAKE_CMD = os.environ.get("QTS_GPU_WAKE_CMD", "")
# Canonical REGISTRY name -- see the note in sentiment_gpu_job.py. Keep these two in step.
MODEL_NAME = "qwen3:14b"


@contextmanager
def corpus_lock(timeout_s: int = 1800):
    ensure_dirs()
    f = open(OUTPUTS_DIR / ".qts-corpus.lock", "w")
    waited = 0
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if waited >= timeout_s:
                f.close()
                raise TimeoutError("corpus lock timeout")
            time.sleep(5)
            waited += 5
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ssh_ok() -> bool:
    return subprocess.run([*_SSH, "true"], capture_output=True).returncode == 0


def _wake_and_wait(deadline_check, wake_budget_s: int = MAX_WAKE_WAIT_S) -> bool:
    """WoL + poll SSH until reachable. Returns False (skip cleanly) if the GPU worker is not reachable
    within `wake_budget_s`, or if the run deadline hits first. The bounded budget is what makes
    this polite: when the user is on the box (other OS -> ssh closed), we give up in ~3 min
    instead of spinning, and the run exits 0 for the next slot to retry."""
    if _ssh_ok():
        return True
    if GPU_WAKE_CMD:
        subprocess.run(GPU_WAKE_CMD.split(), capture_output=True)
    waited = 0
    while not deadline_check() and waited < wake_budget_s:
        time.sleep(WAKE_POLL_S)
        waited += WAKE_POLL_S
        if _ssh_ok():
            return True
    return False


def _remote_model_present() -> bool:
    """True if GPU worker's Ollama actually has MODEL_NAME.

    Cheap insurance against the 2026-07-11 class of failure: the chat-site model refresh
    deleted the model out from under us and we shipped 10k-text batches at a box that could
    not score a single one, for 8 days, while logging it as a benign skip.
    """
    r = subprocess.run([*_SSH, "curl -s --max-time 15 http://localhost:11434/api/tags"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    try:
        names = [m.get("name", "") for m in json.loads(r.stdout).get("models", [])]
    except (ValueError, AttributeError):
        return False
    base = MODEL_NAME.split(":", 1)[0]
    return any(n == MODEL_NAME or n.split(":", 1)[0] == base for n in names)


class RemoteJobError(RuntimeError):
    """The remote job ran to completion and exited non-zero."""


def _run_remote(records: list[dict], deadline_check) -> list[dict]:
    """Ship batch, run detached, poll DONE; on deadline abort kill the job and
    return whatever partial output exists (the gpu job writes incrementally).

    Raises RemoteJobError if the job finished on its own with a non-zero exit. A deadline
    abort is *expected* to exit non-zero (we killed it), so STATUS is only authoritative
    when the job completed by itself.
    """
    inp = OUTPUTS_DIR / "sentiment_input.json"
    outp = OUTPUTS_DIR / "sentiment_output.json"
    inp.write_text(json.dumps(records))
    job = str(Path(__file__).with_name("sentiment_gpu_job.py"))
    # Clear STATUS and output.json too -- not just DONE. A stale output.json from a previous
    # run would otherwise be scp'd back and silently re-imported as if it were this batch.
    subprocess.run([*_SSH, f"mkdir -p {REMOTE_DIR} && "
                           f"rm -f {REMOTE_DIR}/DONE {REMOTE_DIR}/STATUS "
                           f"{REMOTE_DIR}/output.json"],
                   check=True)
    subprocess.run([*_SCP, job,
                    f"{SSH_USER}@{GPU_HOST}:{REMOTE_DIR}/sentiment_gpu_job.py"],
                   check=True)
    subprocess.run([*_SCP, str(inp),
                    f"{SSH_USER}@{GPU_HOST}:{REMOTE_DIR}/input.json"], check=True)
    # Record the job's exit code BEFORE signalling DONE. Previously `touch DONE` ran
    # unconditionally, so a crashed job was indistinguishable from a successful one.
    subprocess.run([*_SSH,
                    f"cd {REMOTE_DIR} && nohup bash -c "
                    f"'python3 sentiment_gpu_job.py input.json output.json "
                    f"> job.log 2>&1; echo $? > STATUS; touch DONE' >/dev/null 2>&1 "
                    f"& echo started"],
                   check=True)
    finished = False
    while not deadline_check():
        time.sleep(JOB_POLL_S)
        done = subprocess.run([*_SSH, f"test -f {REMOTE_DIR}/DONE && echo yes"],
                              capture_output=True, text=True).stdout.strip()
        if done == "yes":
            finished = True
            break
    if not finished:
        subprocess.run([*_SSH, "pkill -f sentiment_gpu_job.py"], check=False)
        print("! deadline hit — job aborted, importing partial output")
    else:
        status = subprocess.run([*_SSH, f"cat {REMOTE_DIR}/STATUS 2>/dev/null"],
                                capture_output=True, text=True).stdout.strip()
        if status not in ("0", ""):
            tail = subprocess.run([*_SSH, f"tail -5 {REMOTE_DIR}/job.log 2>/dev/null"],
                                  capture_output=True, text=True).stdout.strip()
            raise RemoteJobError(f"remote job exited {status}; job.log tail:\n{tail}")
    r = subprocess.run([*_SCP, f"{SSH_USER}@{GPU_HOST}:{REMOTE_DIR}/output.json",
                        str(outp)], capture_output=True)
    if r.returncode != 0 or not outp.exists():
        return []
    return json.loads(outp.read_text())


def run(now=_utcnow, corpus_path=None, max_run_minutes: int = MAX_RUN_MINUTES) -> dict:
    # Per-RUN deadline (elapsed since this slot started), not an absolute wall-clock window.
    start = now()

    def past_deadline() -> bool:
        return (now() - start).total_seconds() >= max_run_minutes * 60

    kwargs = {} if corpus_path is None else {"path": corpus_path}
    with corpus_lock(), CorpusStore(read_only=True, **kwargs) as c:
        texts = c.unscored_texts(BATCH_CAP)
    if texts.empty:
        print("sentiment: no unscored texts")
        return {"skipped": "no_backlog"}
    print(f"sentiment: {len(texts)} texts to score")

    if not _wake_and_wait(past_deadline):
        print("sentiment: GPU worker unavailable — skipping this slot (will retry next)")
        return {"skipped": "gpu_unavailable"}

    # An unreachable box is a benign skip; a reachable box that cannot score is a FAULT.
    # Conflating the two is what let 8 days of empty runs look routine.
    if not _remote_model_present():
        print(f"sentiment: FAILED — model {MODEL_NAME!r} is not present on the GPU worker. "
              f"Nothing was scored; re-pull it (`ollama pull {MODEL_NAME}`) before the "
              f"next slot.")
        return {"failed": "model_missing", "model": MODEL_NAME}

    try:
        feats = _run_remote(texts.to_dict("records"), past_deadline)
    except RemoteJobError as e:
        print(f"sentiment: FAILED — {e}")
        return {"failed": "job_error", "detail": str(e)}
    if not feats:
        print(f"sentiment: FAILED — job returned 0 scores for {len(texts)} texts. "
              f"This is an infrastructure fault, not an empty backlog.")
        return {"failed": "no_scores", "requested": int(len(texts))}
    df = pd.DataFrame(feats)
    df["model"] = MODEL_NAME
    df["scored_at"] = pd.Timestamp(now().replace(tzinfo=None))
    with corpus_lock(), CorpusStore(**kwargs) as c:
        n = c.upsert_sentiment(df[["text_id", "value", "model", "scored_at"]])
    print(f"sentiment: imported {n} scores "
          f"({len(texts) - n} re-queue for next window)")
    return {"scored": n}


if __name__ == "__main__":
    # Exit non-zero on a fault so cron/monitoring can distinguish "nothing to do" from
    # "this has been broken for a week".
    result = run()
    raise SystemExit(1 if "failed" in result else 0)
