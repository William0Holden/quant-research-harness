#!/usr/bin/env bash
# deploy/reddit-collector.sh — twice-hourly Reddit subreddit sweep into the corpus DB (never qts.duckdb).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env ] && . .env; set +a
LOG=outputs/reddit.log
{
  exec 9>"outputs/.qts-corpus.lock"
  flock -w 1800 9 || { echo "corpus lock timeout $(date -Is)"; exit 1; }
  echo "=== reddit $(date -Is) ==="
  uv run python -m qts.altdata.reddit
} >> "$LOG" 2>&1
