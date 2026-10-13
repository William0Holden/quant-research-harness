#!/usr/bin/env bash
# Crypto sentiment collection: crypto subreddits + crypto news RSS -> shared corpus, linked
# to coins. The existing gpu sentiment-scorer + altdata.aggregate then flow it into
# alt_signals with no further work. Runs hourly like the equity collectors.
#
# GATED: everything this feeds is a RISK signal that touches NOTHING (sizing, execution)
# until it earns un-gating on its forward risk-IC record. See the 2026-07-19 design doc.
#
#   20 * * * * <REPO_DIR>/deploy/crypto-sentiment-collect.sh  # qts-crypto
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH"
LOG=outputs/logs/crypto-sentiment-collect.log
mkdir -p "$(dirname "$LOG")"
# Shell-level flock on the corpus, matching the equity collectors (news/reddit share the DB).
{
  echo "=== $(date -u +%FT%TZ) crypto sentiment collect ==="
  flock -w 120 outputs/.qts-corpus.lock uv run python -m qts.crypto.sentiment.collect
} >> "$LOG" 2>&1
