"""Unit tests for the pure logic: parsing, normalisation, dedup and upsert."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import extract_skill_flags  # noqa: E402
from database import JobListing, compute_job_hash, count_jobs, upsert_jobs  # noqa: E402
from scraper import (  # noqa: E402
    JobParser, categorize_role, clean_company, normalize_experience,
    normalize_location, parse_salary, run_scrape,
)


@pytest.mark.parametrize("raw, expected", [
    ("$150k - $190k", (150_000, 190_000)),
    ("150,000 - 190,000 USD", (150_000, 190_000)),
    ("$175-200k", (175_000, 200_000)),
    ("$72 - $91 per hour", (149_760, 189_280)),
    ("USD 120K-160K / yr", (120_000, 160_000)),
    ("$140,000", (140_000, 140_000)),
    ("Competitive salary", (None, None)),
    ("", (None, None)),
    ("$5 - $9", (None, None)),  # implausibly low -> rejected, not mis-parsed
    ("\u20ac60,000 - \u20ac80,000", (None, None)),  # EUR must never be read as USD
    ("\u00a350k - \u00a360k", (None, None)),
])
def test_parse_salary(raw, expected):
    assert parse_salary(raw) == expected


def test_hash_ignores_case_punctuation_and_legal_suffix():
    a = compute_job_hash("Acme, Inc.", "Senior Data Scientist")
    assert a == compute_job_hash("ACME", "senior  data scientist")
    assert a != compute_job_hash("Acme", "Data Scientist")


def test_normalisers():
    assert normalize_location("SF Bay Area") == "San Francisco"
    assert normalize_location("NYC (Hybrid)") == "New York"
    assert normalize_location("Anywhere (Remote)") == "Remote"
    assert normalize_location("Mars Colony") == "Other"
    assert categorize_role("Sr. MLOps Engineer") == "ML Engineer"
    assert categorize_role("Analytics Engineer III") == "Data Engineer"
    assert categorize_role("Barista") == "Other"
    assert normalize_experience("Staff Data Scientist", "") == "Lead"
    assert normalize_experience("Data Scientist", "We need 6+ years of experience") == "Senior"
    assert clean_company("  NORTHWIND ANALYTICS, Inc. ") == "Northwind Analytics"


def test_skill_regex_boundaries():
    flags = extract_skill_flags("Experience with NoSQL, JavaScript and large language models")
    assert flags["SQL"] == 0 and flags["Java"] == 0 and flags["LLMs"] == 1


def test_parser_handles_all_layouts_and_drops_malformed():
    html = """
    <div class="job-card"><h2 class="job-title">Senior Data Engineer</h2>
      <span class="company">Acme Inc.</span><span class="loc">NYC</span>
      <span class="pay">$150k - $190k</span><div class="desc">Spark &amp; Airflow, 5+ years</div></div>
    <li class="result-row"><a class="title-link">ML Engineer</a>
      <div class="meta"><b>Beta LLC</b> &bull; <i>Remote</i></div>
      <p class="snippet">PyTorch. Compensation: $140k - $170k</p></li>
    <article class="posting"><h3 class="posting-title">Data Scientist</h3>
      <span class="org"></span><section class="body">no company</section></article>
    """
    listings, dropped, cards = JobParser("test").parse_page(html)
    assert (len(listings), dropped, cards) == (2, 1, 3)
    assert listings[0].salary_mid == 170_000 and listings[0].company == "Acme"
    assert listings[1].salary_mid == 155_000  # recovered from the description text


def test_upsert_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    job = JobListing("Data Scientist", "Acme", "Remote", "Python", "Mid", "Data Scientist",
                     "$100k - $120k", 100_000, 120_000, "test")
    assert upsert_jobs([job], db).inserted == 1
    r = upsert_jobs([job, job], db)
    assert (r.inserted, r.updated) == (0, 2) and count_jobs(db) == 1


def test_end_to_end_scrape_is_reproducible(tmp_path):
    s1 = run_scrape(pages=3, seed=7, db_path=tmp_path / "a.db", failure_rate=0.0)
    s2 = run_scrape(pages=3, seed=7, db_path=tmp_path / "a.db", failure_rate=0.0)
    assert s1.inserted > 0 and s2.inserted == 0 and s2.updated == s2.listings_parsed


# --------------------------------------------------------------------------- #
# Real-data source (Remotive), tested against a local server that serves the
# documented API schema - exercises the real HTTP, cache, retry and parse code.
# --------------------------------------------------------------------------- #
import json  # noqa: E402
import sqlite3  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

import numpy as np  # noqa: E402

import scraper  # noqa: E402
from database import fetch_jobs_df, get_connection  # noqa: E402
from model import (  # noqa: E402
    FEATURE_COLUMNS, fit_interval_models, interval_bounds, load_model, load_training_frame,
    predict_with_interval, train_and_save,
)
from scheduler import safe  # noqa: E402
from scraper import parse_remotive_job  # noqa: E402

FIXTURE = {
    "0-legal-notice": "Remotive API Legal Notice",
    "job-count": 4,
    "jobs": [
        {"id": 1, "url": "https://remotive.com/remote-jobs/software-dev/senior-backend-engineer-1",
         "title": "Senior Backend Engineer", "company_name": "Acme Inc.", "category": "Software Development",
         "salary": "$120,000 - $150,000", "candidate_required_location": "USA Only",
         "description": "<p>Python &amp; <b>AWS</b>, 6+ years</p>"},
        {"id": 2, "url": "https://remotive.com/remote-jobs/data/ml-engineer-2", "title": "Machine Learning Engineer",
         "company_name": "Beta", "salary": "\u20ac70,000 - \u20ac90,000", "description": "<p>PyTorch and LLMs</p>"},
        {"id": 3, "url": "https://remotive.com/remote-jobs/marketing/seo-3", "title": "SEO Manager",
         "company_name": "Gamma", "salary": "", "description": "<p>Content</p>"},
        {"id": 4, "title": "Data Engineer", "company_name": "", "description": "no company"},
    ],
}


@pytest.fixture()
def fake_remotive():
    """Local HTTP server mimicking the Remotive endpoint; counts requests."""
    state = {"hits": 0, "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            state["hits"] += 1
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(FIXTURE).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/api/remote-jobs", state
    server.shutdown()


def test_parse_remotive_job_filters_and_cleans():
    jobs = [parse_remotive_job(j) for j in FIXTURE["jobs"]]
    first, ml, seo, missing = jobs
    assert first.company == "Acme" and first.role_category == "Software Engineer"
    assert first.salary_mid == 135_000 and first.experience_level == "Senior"
    assert "Python" in first.description and "<" not in first.description  # HTML stripped
    assert first.url.startswith("https://remotive.com/") and first.source == "remotive"
    assert ml.salary_mid is None            # EUR salary rejected, listing itself kept
    assert seo is None and missing is None  # irrelevant role / missing company dropped


def test_remotive_pipeline_caches_and_attributes(fake_remotive, tmp_path):
    url, state = fake_remotive
    kwargs = dict(source="remotive", db_path=tmp_path / "r.db", remotive_url=url,
                  cache_dir=tmp_path / "cache", min_delay=0, max_delay=0)
    first = scraper.run_scrape(**kwargs)
    assert (first.inserted, first.pages_ok, state["hits"]) == (2, 1, 1)
    second = scraper.run_scrape(**kwargs)           # served from cache: NO extra request
    assert state["hits"] == 1 and second.inserted == 0 and second.updated == 2
    df = fetch_jobs_df(tmp_path / "r.db")
    assert set(df["source"]) == {"remotive"} and df["url"].notna().all()


def test_remotive_failure_is_graceful(fake_remotive, tmp_path, monkeypatch):
    url, state = fake_remotive
    state["status"] = 500
    monkeypatch.setattr(scraper.time, "sleep", lambda s: None)
    summary = scraper.run_scrape(source="remotive", db_path=tmp_path / "f.db", remotive_url=url,
                                 cache_dir=tmp_path / "c", min_delay=0, max_delay=0)
    assert summary.pages_failed == 1 and summary.inserted == 0 and state["hits"] == 3  # 3 retries


def test_schema_migration_adds_url_column(tmp_path):
    """A database created by the previous version (no `url` column) is upgraded in place."""
    import database
    old_schema = database.SCHEMA_SQL.replace("    url              TEXT,\n", "")
    assert "url" not in old_schema.split("scrape_runs")[0]
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(old_schema)
        conn.execute("INSERT INTO jobs (job_hash, title, company, first_seen, last_seen) "
                     "VALUES ('h', 't', 'c', 'now', 'now')")
    with get_connection(db) as conn:
        assert "url" in {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1  # data preserved


# --------------------------------------------------------------------------- #
# Model: conformal intervals, tracking-free training, scheduler isolation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sim_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("sim") / "sim.db"
    run_scrape(pages=24, seed=3, db_path=db, failure_rate=0.0)
    return db


def test_conformal_interval_achieves_nominal_coverage(sim_db):
    from sklearn.model_selection import train_test_split
    df = load_training_frame(sim_db, "simulated")
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURE_COLUMNS], df["salary_mid"], test_size=0.3, random_state=0)
    models = fit_interval_models(X_tr, y_tr, alpha=0.2)
    lo, hi = interval_bounds(models, X_te)
    coverage = float(np.mean((y_te.to_numpy() >= lo) & (y_te.to_numpy() <= hi)))
    assert (lo <= hi).all() and 0.72 <= coverage <= 0.96


def test_train_save_load_predict_roundtrip(sim_db, tmp_path):
    metrics = train_and_save(sim_db, tmp_path / "m.joblib", tmp_path / "m.json", min_rows=50)
    assert metrics["r2"] > 0.8 and metrics["trained_on"] == "simulated"
    assert metrics["interval_coverage"] > metrics["naive_mae_band_coverage"]  # CQR beats +/-MAE band
    bundle = load_model(tmp_path / "m.joblib")
    point, low, high = predict_with_interval(bundle, "ML Engineer", "San Francisco", "Senior", ["Python", "LLMs"])
    assert 0 < low <= point <= high
    junior, _, _ = predict_with_interval(bundle, "ML Engineer", "San Francisco", "Junior", ["Python"])
    assert junior < point  # seniority + skills must raise the estimate


def test_training_refuses_tiny_or_wrong_source(sim_db, tmp_path):
    with pytest.raises(ValueError):
        train_and_save(sim_db, tmp_path / "x.joblib", tmp_path / "x.json", source="remotive")


def test_scheduler_job_wrapper_swallows_errors():
    def boom():
        raise RuntimeError("nope")
    safe("boom", boom)()  # must not raise
