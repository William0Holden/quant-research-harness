#!/usr/bin/env bash
# deploy/news-collector.sh — hourly RSS/news sweep into the corpus DB (never qts.duckdb).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env ] && . .env; set +a
LOG=outputs/news.log
{
  exec 9>"outputs/.qts-corpus.lock"
  flock -w 1800 9 || { echo "corpus lock timeout $(date -Is)"; exit 1; }
  echo "=== news $(date -Is) ==="
  uv run python -m qts.altdata.news
} >> "$LOG" 2>&1
