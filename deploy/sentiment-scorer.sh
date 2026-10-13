#!/usr/bin/env bash
# Sentiment scorer: score new corpus texts on the GPU worker. RUNS SEVERAL TIMES A DAY (see cron).
#
# CADENCE (user, 2026-07-19): the old 4-8am-only window is relaxed. Each run works for a
# bounded budget (MAX_RUN_MINUTES) and drains one batch; successive slots drain a backlog and
# catch GPU worker whenever it is briefly reachable. Every run SKIPS CLEANLY if the GPU worker is not
# reachable (MAX_WAKE_WAIT_S) -- so the user is never fought for the machine (a box in another OS
# has ssh closed -> unreachable -> skip).
#
# Overlap guard: a non-blocking flock on a DEDICATED lock (NOT the corpus lock) so a slow slot
# never overlaps the next. The corpus lock is taken by the python driver only around export/
# import, so hourly collectors keep running -- do not flock the corpus here.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env ] && . .env; set +a
LOG=outputs/sentiment.log
mkdir -p outputs
exec 9>outputs/.qts-sentiment-scorer.lock
if ! flock -n 9; then
  echo "=== sentiment $(date -Is): a run is already in progress — skipping slot ===" >> "$LOG"
  exit 0
fi
{
  echo "=== sentiment $(date -Is) ==="
  uv run python -m qts.nlp.sentiment_factory
} >> "$LOG" 2>&1
