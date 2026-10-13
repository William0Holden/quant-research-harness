"""Sentiment batch job — RUNS ON GPU worker. Pure stdlib + Ollama HTTP.

Reads a JSON list [{text_id, text}] of short news/social texts; writes a JSON
list [{text_id, value}] with value in [-1, 1]. Skips texts Ollama can't score.
Usage (on the GPU worker):  python3 sentiment_gpu_job.py input.json output.json
"""
import json
import re
import sys
import urllib.request

OLLAMA = "http://localhost:11434"
# Canonical REGISTRY name, deliberately not a local alias. The old local alias `qwen3-14b`
# was deleted by the 2026-07-11 chat-site model refresh and every request 404'd for 8 days;
# a registry name can always be re-pulled. Scores are reproducible across a re-pull because
# we call with temperature 0 -- only the weights matter, not num_ctx or sampling params.
GEN_MODEL = "qwen3:14b"
# If this many attempts pass with nothing scored, the fault is infrastructural (missing model,
# Ollama down) rather than a run of unparseable texts. Abort instead of burning the batch.
ZERO_PROGRESS_ABORT = 20

PROMPT = ("You are a financial analyst. Rate the sentiment of this financial "
          "news or social-media text toward the company/companies it mentions, "
          "from -1.0 (very bearish) to 1.0 (very bullish). Reply with ONLY the "
          "number.\n\nText:\n")


def _post(path, payload, timeout=120):
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def score(text):
    try:
        r = _post("/api/generate", {
            "model": GEN_MODEL, "prompt": PROMPT + text[:2500], "stream": False,
            "think": False, "options": {"num_predict": 12, "temperature": 0}})
        m = re.search(r"-?\d+(?:\.\d+)?", r.get("response", ""))
        return max(-1.0, min(1.0, float(m.group()))) if m else None
    except Exception:
        return None


def _tags():
    """Model names Ollama currently has. Separate function so tests can stub it."""
    with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=30) as r:
        return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]


def preflight():
    """Refuse to start unless the model is actually present.

    Without this, a missing model means 10,000 silently-swallowed 404s and an empty output
    file that every downstream layer reads as 'nothing to do'. Exits non-zero so the caller
    can tell a broken box from an idle one.
    """
    try:
        names = _tags()
    except Exception as e:  # noqa: BLE001 - any failure to reach Ollama is fatal here
        sys.exit(f"FATAL: cannot reach Ollama at {OLLAMA}: {e}")
    # Ollama may report a tagged name (`qwen3:14b`); match exactly or on the bare name.
    if not any(n == GEN_MODEL or n.split(":", 1)[0] == GEN_MODEL.split(":", 1)[0]
               for n in names):
        sys.exit(f"FATAL: model {GEN_MODEL!r} not present on this host. Have: {names}")


def main(inp, outp):
    preflight()
    texts = json.load(open(inp))
    out = []
    for i, t in enumerate(texts):
        v = score(t.get("text") or "")
        if v is not None:
            out.append({"text_id": t["text_id"], "value": v})
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            print(f"[{i + 1}/{len(texts)}] scored={len(out)}", flush=True)
        # incremental output so a deadline abort still yields partial results
        json.dump(out, open(outp, "w"))
        if i + 1 >= ZERO_PROGRESS_ABORT and not out:
            sys.exit(f"FATAL: {i + 1} attempts, 0 scored — treating as infrastructure "
                     f"failure rather than scoring the remaining {len(texts) - i - 1} texts")
    json.dump(out, open(outp, "w"))
    print(f"wrote {len(out)} sentiment rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
