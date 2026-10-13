#!/usr/bin/env bash
# deploy/daily-rebalance.sh — Mon–Thu post-pipeline rebalance. GATED OFF by default:
# runs only when QTS_DAILY_REBALANCE=1 is set in .env. Flip it on only after a fast
# signal has earned un-gating on the Signal IC record (see plan/README). Friday stays
# weekly-rebalance.sh. Orders queue to next market open (next-bar discipline).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env ] && . .env; set +a
[ "${QTS_DAILY_REBALANCE:-0}" = "1" ] || exit 0
LOG=outputs/rebalance.log
{
  exec 9>"outputs/.qts-db.lock"
  flock -w 5400 9 || { echo "lock timeout"; exit 1; }
  echo "=== daily rebalance $(date -Is) ==="
  uv run python -m qts.exec.rebalance --submit --cadence daily
} >> "$LOG" 2>&1
