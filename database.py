"""SQLite persistence layer with idempotent upserts.

Deduplication key: SHA-256 of the *normalised* ``company + job title``.
Re-scraping the same posting therefore updates the existing row instead of
creating a duplicate, no matter how many times the pipeline runs.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)

_COMPANY_SUFFIXES = {"inc", "llc", "ltd", "corp", "corporation", "co", "company", "plc", "gmbh"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_hash         TEXT    NOT NULL UNIQUE,
    title            TEXT    NOT NULL,
    company          TEXT    NOT NULL,
    location         TEXT,
    description      TEXT,
    experience_level TEXT,
    role_category    TEXT,
    salary_raw       TEXT,
    salary_min       REAL,
    salary_max       REAL,
    salary_mid       REAL,
    source           TEXT,
    url              TEXT,
    first_seen       TEXT    NOT NULL,
    last_seen        TEXT    NOT NULL,
    times_seen       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_jobs_role     ON jobs (role_category);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs (location);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT    NOT NULL,
    finished_at       TEXT    NOT NULL,
    source            TEXT,
    pages_requested   INTEGER,
    pages_ok          INTEGER,
    pages_failed      INTEGER,
    listings_parsed   INTEGER,
    listings_dropped  INTEGER,
    inserted          INTEGER,
    updated           INTEGER
);
"""

# ON CONFLICT ... DO UPDATE requires SQLite >= 3.24 (bundled with Python 3.8+).
# Latest posting wins: every mutable column is refreshed; first_seen is kept.
UPSERT_SQL = """
INSERT INTO jobs (
    job_hash, title, company, location, description, experience_level,
    role_category, salary_raw, salary_min, salary_max, salary_mid, source, url,
    first_seen, last_seen, times_seen
) VALUES (
    :job_hash, :title, :company, :location, :description, :experience_level,
    :role_category, :salary_raw, :salary_min, :salary_max, :salary_mid, :source, :url,
    :now, :now, 1
)
ON CONFLICT(job_hash) DO UPDATE SET
    title            = excluded.title,
    company          = excluded.company,
    location         = excluded.location,
    description      = excluded.description,
    experience_level = excluded.experience_level,
    role_category    = excluded.role_category,
    salary_raw       = excluded.salary_raw,
    salary_min       = excluded.salary_min,
    salary_max       = excluded.salary_max,
    salary_mid       = excluded.salary_mid,
    source           = excluded.source,
    url              = excluded.url,
    last_seen        = excluded.last_seen,
    times_seen       = jobs.times_seen + 1;
"""


@dataclass(frozen=True)
class JobListing:
    """A cleaned, normalised job posting ready for persistence."""

    title: str
    company: str
    location: str
    description: str
    experience_level: str
    role_category: str
    salary_raw: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    source: str
    url: Optional[str] = None  # canonical link back to the original posting (attribution)

    @property
    def salary_mid(self) -> Optional[float]:
        """Midpoint of the annualised salary range (``None`` if undisclosed)."""
        if self.salary_min is None or self.salary_max is None:
            return None
        return (self.salary_min + self.salary_max) / 2.0


@dataclass(frozen=True)
class UpsertResult:
    """Counts returned by :func:`upsert_jobs`."""

    inserted: int
    updated: int


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalise(text: str, strip_suffixes: bool = False) -> str:
    tokens = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
    if strip_suffixes:
        while tokens and tokens[-1] in _COMPANY_SUFFIXES:
            tokens.pop()
    return " ".join(tokens)


def compute_job_hash(company: str, title: str) -> str:
    """Stable dedup key: 'Acme, Inc.' and 'ACME' + same title hash identically."""
    key = f"{_normalise(company, strip_suffixes=True)}|{_normalise(title)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply lightweight, idempotent schema migrations for databases created by older versions."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "url" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN url TEXT")
        logger.info("Migrated database: added jobs.url")


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a transactional connection; commit on success, rollback on error.

    The schema is created on first use, so callers never need to worry about
    initialisation order (important for the Streamlit app on a fresh volume).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")  # readers don't block the writer
        conn.executescript(SCHEMA_SQL)
        _migrate(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction failed; rolled back")
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database file and schema if they do not exist."""
    with get_connection(db_path):
        logger.info("Database ready at %s", db_path)


def upsert_jobs(jobs: Iterable[JobListing], db_path: Path = DB_PATH) -> UpsertResult:
    """Insert new listings and update existing ones, atomically.

    Returns:
        Number of rows inserted and updated.
    """
    now = utc_now()
    inserted = updated = 0
    with get_connection(db_path) as conn:
        for job in jobs:
            job_hash = compute_job_hash(job.company, job.title)
            exists = (
                conn.execute("SELECT 1 FROM jobs WHERE job_hash = ?", (job_hash,)).fetchone()
                is not None
            )
            conn.execute(
                UPSERT_SQL,
                {
                    "job_hash": job_hash,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description,
                    "experience_level": job.experience_level,
                    "role_category": job.role_category,
                    "salary_raw": job.salary_raw,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "salary_mid": job.salary_mid,
                    "source": job.source,
                    "url": job.url,
                    "now": now,
                },
            )
            if exists:
                updated += 1
            else:
                inserted += 1
    logger.info("Upsert complete: %d inserted, %d updated", inserted, updated)
    return UpsertResult(inserted=inserted, updated=updated)


def log_scrape_run(
    started_at: str,
    source: str,
    pages_requested: int,
    pages_ok: int,
    pages_failed: int,
    listings_parsed: int,
    listings_dropped: int,
    result: UpsertResult,
    db_path: Path = DB_PATH,
) -> None:
    """Persist run-level statistics (powers the dashboard's pipeline history)."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO scrape_runs (started_at, finished_at, source, pages_requested,
               pages_ok, pages_failed, listings_parsed, listings_dropped, inserted, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                started_at, utc_now(), source, pages_requested, pages_ok, pages_failed,
                listings_parsed, listings_dropped, result.inserted, result.updated,
            ),
        )


def fetch_jobs_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load every job posting into a DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM jobs ORDER BY last_seen DESC, id DESC", conn)


def fetch_scrape_runs_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load scrape-run history, newest first."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM scrape_runs ORDER BY id DESC", conn)


def count_jobs(db_path: Path = DB_PATH) -> int:
    """Total number of unique job postings stored."""
    with get_connection(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
