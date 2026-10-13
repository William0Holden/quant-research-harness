#!/usr/bin/env bash
# Launch the dashboard with env (incl. QTS_DASH_PASSWORD) loaded from .env.
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . .env; set +a
exec uv run streamlit run src/qts/dashboard/app.py \
    --server.address 0.0.0.0 --server.port 8501 --server.headless true \
    --server.fileWatcherType none --browser.gatherUsageStats false
