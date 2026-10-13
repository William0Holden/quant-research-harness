#!/usr/bin/env bash
# deploy/daily-pipeline.sh — data pipeline around the DuckDB single-writer rule.
#   (no flags)   prices only, incremental          (Mon–Thu nightly)
#   --with-sec   + SEC fundamentals/events         (Fri pre-rebalance, Sun)
#   --with-nlp   + 10-K filings ingest + GPU worker NLP factory (Sun)
# If the dashboard runs as a systemd user unit named qts-dashboard it holds a read-only
# handle; a writer needs the exclusive lock, so we
# stop the service, write, and restart it even if a step fails.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
# cron has no session bus: without these, every systemctl --user below fails
# ("Failed to connect to bus") and the dashboard never actually stops -- fine
# while the dashboard held no persistent DB handle, fatal once one does
# (2026-07-10: preview dashboard held a read handle; whole write phase failed).
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
set -a; [ -f .env ] && . .env; set +a

WITH_SEC=0; WITH_NLP=0
for a in "$@"; do
  case "$a" in
    --with-sec) WITH_SEC=1 ;;
    --with-nlp) WITH_SEC=1; WITH_NLP=1 ;;
  esac
done

LOG=outputs/pipeline.log
{
  exec 9>"outputs/.qts-db.lock"
  flock 9
  echo "=== pipeline start $(date -Is) sec=$WITH_SEC nlp=$WITH_NLP ==="
  systemctl --user stop qts-dashboard 2>/dev/null || true
  sleep 2

  if [ "$WITH_SEC" = 1 ]; then
    uv run python -m qts.ingest.run --incremental
  else
    uv run python -m qts.ingest.run --incremental --no-sec
  fi

  # distill alt-data corpus -> alt_signals (main-DB write; corpus read under its lock)
  (
    exec 8>"outputs/.qts-corpus.lock"
    flock -w 1800 8 && uv run python -m qts.altdata.aggregate
  ) || echo "! aggregate failed"

  if [ "$WITH_SEC" = 1 ]; then
    uv run python -m qts.research.ic_snapshot || echo "! ic snapshot failed"
  fi

  if [ "$WITH_NLP" = 1 ]; then
    uv run python -m qts.ingest.filings || echo "! filings ingest failed"
    uv run python -m qts.nlp.factory   || echo "! nlp factory failed"
  fi

  systemctl --user start qts-dashboard 2>/dev/null || true
  echo "=== pipeline done $(date -Is) ==="
} >> "$LOG" 2>&1
