#!/usr/bin/env bash
# Crypto trend book rebalance.
#
# NOT SCHEDULED YET. This needs a SECOND Alpaca paper account:
#   app.alpaca.markets -> click the paper account number (upper-left) -> "Open New Paper Account"
#   then put its keys in .env as APCA_CRYPTO_KEY_ID / APCA_CRYPTO_SECRET_KEY.
# Until then `profile.guard_account()` refuses to run -- crypto and equities share ONE
# Alpaca account, so without separate accounts each book would liquidate the other.
#
# When it IS scheduled, the window is Tue-Thu 14:00-20:00 UTC (US-hours liquidity) --
# deliberately NOT the equity book's Friday 23:30 UTC, which in crypto is Friday night, the
# thinnest book of the week (weekend volume -20-25%, costs +11%).
#
#   0 16 * * 2-4  <REPO_DIR>/deploy/crypto-rebalance.sh  # qts-crypto
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
export PATH="$HOME/.local/bin:$PATH"

LOG=outputs/logs/crypto-rebalance.log
mkdir -p "$(dirname "$LOG")"
# NOTE: no `tee -a` here. The cron entry appends with `>>`; adding tee would double every
# line (see the watchdog dup-log bug of 2026-05-17).
{
  echo "=== $(date -u +%FT%TZ) crypto rebalance ==="
  uv run python -m qts.crypto.rebalance "${@:---live}"
} >> "$LOG" 2>&1
