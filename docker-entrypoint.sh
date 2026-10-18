#!/bin/sh
# Bootstraps data + model on first start, then launches the dashboard.
# Mount ./data and ./models as volumes to persist state across container restarts.
set -e

if [ ! -f "data/jobs.db" ]; then
    echo "[entrypoint] No database found - scraping initial snapshot..."
    python scraper.py --pages 40 --seed 42
fi

if [ ! -f "models/salary_model.joblib" ]; then
    echo "[entrypoint] No model found - training..."
    python model.py
fi

exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
