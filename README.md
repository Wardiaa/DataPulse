# Real-Time Tech Job Market Intelligence Engine

End-to-end data product: **scrape -> clean -> store (upsert) -> feature-engineer -> train -> serve**.
Tracks Data Scientist, Data Engineer, ML Engineer and Software Engineer postings, surfaces
market KPIs, and predicts market-rate salary from role, location, seniority and skills.

## Architecture

```
 +----------------+   HTML    +------------------+  JobListing  +-----------------+
 | Fetcher        | --------> | JobParser        | -----------> | database.py     |
 |  Simulated |   |           |  multi-selector  |              |  SQLite + hash  |
 |  Http (polite) |           |  salary/location |              |  UPSERT (WAL)   |
 +----------------+           |  normalisation   |              +--------+--------+
   retry + backoff            +------------------+                       |
   rate limit + UA rotation                                              v
                                                    +--------------------+------------------+
                                                    | model.py                              |
                                                    |  regex skill flags -> OneHot + flags  |
                                                    |  -> XGBoost (sklearn Pipeline)        |
                                                    |  hold-out + 5-fold CV -> joblib       |
                                                    +--------------------+------------------+
                                                                         |
                                                    +--------------------v------------------+
                                                    | app.py (Streamlit)                    |
                                                    |  live KPIs . predictor . model card   |
                                                    +---------------------------------------+
```

| File | Responsibility |
|---|---|
| `config.py` | Paths, taxonomies, skill regex dictionary, logging setup |
| `database.py` | Schema, SHA-256 dedup key, atomic upsert, run history, loaders |
| `scraper.py` | Simulator + HTML parser, Remotive real-data client, retry/rate-limit/cache, CLI |
| `model.py` | Skill flags, pipeline, CQR intervals, evaluation, MLflow hook, inference |
| `scheduler.py` | APScheduler: periodic scrapes + conditional retraining |
| `app.py` | Dashboard + salary calculator + "next best skill" analysis |
| `tests/` | 25 unit + integration tests |
| `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `Makefile` | Packaging, orchestration, CI |

## Quick start

```bash
pip install -r requirements.txt
python scraper.py --pages 40 --seed 42     # reproducible simulated snapshot (3 messy HTML layouts)
python model.py                            # RMSE / MAE / R2 / interval coverage -> models/
streamlit run app.py                       # http://localhost:8501  (also bootstraps itself if empty)
python -m pytest tests                     # needs: pip install pytest
```

Shortcuts are in the `Makefile` (`make data model run test`).

### Real data (Remotive public API)

```bash
python scraper.py --source remotive        # ONE request, cached 6 h
python model.py --source remotive --min-rows 30    # train on real postings (if enough disclose salary)
```
Or use **Fetch real data** in the dashboard sidebar and flip **Market view** to *Remotive (real)*.
Remotive's terms are built in: max ~4 fetches/day (6 h response cache), >2 req/min is blocked
(single request, randomised delay), **attribution + link back** to every posting (stored in `jobs.url`,
rendered in the dashboard), and listings are never forwarded to other boards. Real and simulated rows are
**never mixed for training** (`--source` selects one) because that would contaminate the target.

### Automation and tracking

```bash
python scheduler.py                        # simulator every 30 min, Remotive every 6 h, retrain daily (if new rows)
python scheduler.py --run-once --no-remotive
pip install mlflow && python model.py --track && mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Jobs are isolated (a failure is logged, never fatal) and `max_instances=1` prevents overlapping runs.

### Docker

```bash
docker compose up --build                  # dashboard :8501 + scheduler sharing named volumes
docker compose --profile tracking up       # + MLflow UI on :5000
# or just the dashboard:
docker build -t job-intel . && docker run -p 8501:8501 job-intel
```

### Deploy a public demo (free)

*Streamlit Community Cloud*: push this folder to GitHub -> "New app" -> `app.py`. No data or model is committed;
`ensure_ready()` scrapes a seeded snapshot and trains on first start (~10 s). Add the live URL and the CI badge to your CV.

## Prediction intervals (conformalized quantile regression)

A point estimate alone is overconfident, and a "+/- MAE" band is not a real interval. The model fits 10th/90th
percentile XGBoost models and widens them by a margin calibrated on held-out data (CQR), targeting 80 % coverage
with no distributional assumptions. Measured on the untouched test split of the simulated data:

| Band | Target | Measured hold-out coverage |
|---|---|---|
| Naive point estimate +/- MAE | n/a | ~57-59 % |
| **Conformal (CQR)** | 80 % | **~81-84 %** |

(Numbers vary slightly with the seed; both are shown live on the dashboard's *Model & data* tab and logged to MLflow.)

## Design decisions

* **Idempotent ingestion.** Dedup key = SHA-256 of normalised `company|title` (case, punctuation and
  legal suffixes like "Inc." ignored). `INSERT ... ON CONFLICT DO UPDATE`, one transaction per run, WAL mode,
  in-place schema migrations.
* **Resilience.** Per-card exception isolation, exponential-backoff retries, captcha/empty-page detection,
  robots.txt check, randomised delays, rotating User-Agents, on-disk response cache, run statistics persisted.
* **Data hygiene.** Non-USD salaries (EUR/GBP/...) are rejected rather than mis-read as dollars; hourly and
  monthly pay are annualised; implausible values are discarded.
* **No train/serve skew.** Preprocessing lives *inside* the serialised sklearn `Pipeline`;
  `OneHotEncoder(handle_unknown="ignore")` makes unseen categories safe.
* **Honest evaluation.** Hold-out split + 5-fold CV + mean-predictor baseline + interval-coverage check.
  The final artifact is refit on all labelled rows.
* **Interpretability.** Feature importances and a counterfactual "which skill adds the most value" view.
* **Testing.** 25 tests incl. a local HTTP server standing in for the real API, conformal-coverage and
  train/save/load round-trip tests, and a schema-migration test. CI runs tests, a pipeline smoke test and a Docker build.

## Known limitations

* The default data source is a **simulator**. Its salaries come from a known formula plus 5 % noise, so the
  model's high R2 demonstrates the *pipeline*, not real-world predictive power.
* **Real data is thin for salary modelling.** Remotive is a remote-only board, so `location` is always "Remote"
  (the location feature carries no signal), and only a minority of postings disclose salary. Real data is great
  for skill-demand analytics; training on it may need `--min-rows` lowered and will give much weaker metrics.
  The Remotive client was verified against the documented JSON schema using a local stand-in server;
  run `python scraper.py --source remotive` yourself to confirm against the live API.
* Dedup on `company + title` merges same-title postings in different cities (by specification).
* Skill extraction is regex-based: fast and transparent, but misses unlisted synonyms
  (next step: spaCy `PhraseMatcher` or an embedding-based tagger).
* Feature importances / uplift are associational, not causal.

## Extension ideas

Drift monitoring (PSI on skill frequencies), Optuna tuning, FastAPI inference service, Postgres swap-in,
a second real source with disclosed salaries, spaCy skill tagging.
