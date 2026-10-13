"""NLP batch job — RUNS ON GPU worker. Pure stdlib + Ollama HTTP (no torch/transformers).

Reads a JSON list of filings [{symbol, accession, filing_date, period_end, text}], and for
each computes:
  - filing_sentiment: qwen3-14b rates tone/outlook in [-1, 1]  (reasoning tier)
  - an embedding (nomic-embed-text, mean-pooled over chunks) used for the Lazy-Prices diff
Then per symbol (sorted by filing_date) computes lazy_prices_diff = 1 - cosine(prev, cur).

Writes JSON list of {symbol, concept, filing_date, period_end, value}.
Usage (on the GPU worker):  python3 gpu_job.py input.json output.json
"""
import json
import math
import re
import sys
import urllib.request
from collections import defaultdict

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:14b"   # canonical registry name; see sentiment_gpu_job.py


def _post(path, payload, timeout=180):
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(text):
    try:
        return _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text}).get("embedding")
    except Exception:
        return None


def sentiment(text, form):
    prompt = (f"You are a financial analyst. Read this excerpt from a company's SEC Form "
              f"{form} filing (annual report, quarterly report, or earnings press release) "
              "and rate the overall tone and outlook for the company's prospects on a "
              "scale from -1.0 (very negative) to 1.0 (very positive). Reply with ONLY the "
              "number.\n\nExcerpt:\n" + text[:6000])
    try:
        r = _post("/api/generate", {"model": GEN_MODEL, "prompt": prompt, "stream": False,
                                    "think": False, "options": {"num_predict": 12, "temperature": 0}})
        m = re.search(r"-?\d+(?:\.\d+)?", r.get("response", ""))
        if not m:
            return None
        return max(-1.0, min(1.0, float(m.group())))
    except Exception:
        return None


def _chunks(text, n=2000, cap=12):
    return [text[i:i + n] for i in range(0, min(len(text), n * cap), n)]


def _mean(vecs):
    vecs = [v for v in vecs if v]
    if not vecs:
        return None
    L = len(vecs[0])
    out = [0.0] * L
    for v in vecs:
        for i in range(L):
            out[i] += v[i]
    return [x / len(vecs) for x in out]


def _cosine(a, b):
    if not a or not b:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else None


def main(inp, outp):
    filings = json.load(open(inp))
    out = []
    vecs = {}  # (symbol, accession) -> mean embedding
    # (symbol, form) -> [(filing_date, accession, context_only)]. Keyed by accession
    # (not filing_date) because a same-day 8-K press release can share a filing_date
    # with the 10-K/10-Q it accompanies; factory.export_sections already dedupes the
    # *scored* records down to one per (symbol, filing_date), but embeddings here are
    # still looked up per-record via accession to stay unambiguous.
    bysf = defaultdict(list)
    total = len(filings)
    for i, f in enumerate(filings):
        text = f.get("text") or ""
        form = f.get("form") or "10-K"
        context_only = bool(f.get("context_only"))
        s = None if context_only else sentiment(text, form)
        if s is not None:
            out.append({"symbol": f["symbol"], "concept": "filing_sentiment",
                        "filing_date": f["filing_date"], "period_end": f.get("period_end"),
                        "value": s})
        emb = _mean([embed(c) for c in _chunks(text)])
        if emb:
            vecs[(f["symbol"], f["accession"])] = emb
        bysf[(f["symbol"], form)].append((f["filing_date"], f["accession"], context_only))
        print(f"[{i+1}/{total}] {f['symbol']} {f['filing_date']} form={form} "
              f"context_only={context_only} sent={s}", flush=True)

    # Lazy-Prices diff: cosine distance between consecutive same-form filings per symbol.
    # Skip emitting the row when the newer filing is context_only (it's an embedding
    # anchor only, never scored/output).
    for (sym, form), items in bysf.items():
        items = sorted(set(items), key=lambda t: t[0])
        for (prev_date, prev_acc, _prev_ctx), (cur_date, cur_acc, cur_ctx) in zip(items, items[1:]):
            if cur_ctx:
                continue
            a, b = vecs.get((sym, prev_acc)), vecs.get((sym, cur_acc))
            c = _cosine(a, b)
            if c is not None:
                out.append({"symbol": sym, "concept": "lazy_prices_diff",
                            "filing_date": cur_date, "period_end": None, "value": 1.0 - c})

    json.dump(out, open(outp, "w"))
    print(f"wrote {len(out)} feature rows to {outp}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
