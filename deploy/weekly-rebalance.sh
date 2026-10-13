#!/usr/bin/env bash
# deploy/weekly-rebalance.sh — Friday post-close. Reads only (dashboard can stay up).
# With Alpaca keys in .env this SUBMITS to the paper account; without keys the
# runner falls back to DryRunBroker and still writes the audit record.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env ] && . .env; set +a
LOG=outputs/rebalance.log
{
  exec 9>"outputs/.qts-db.lock"
  flock -w 5400 9 || { echo "lock timeout"; exit 1; }
  echo "=== rebalance $(date -Is) ==="
  uv run python -m qts.exec.rebalance --submit
} >> "$LOG" 2>&1
