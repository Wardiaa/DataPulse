"""Shared configuration, constants and helpers for the Job Market Intelligence Engine.

Everything that more than one module needs (paths, taxonomies, the skill
dictionary and logging setup) lives here so there is a single source of truth.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (overridable through environment variables for Docker / CI)
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = Path(os.getenv("JOBS_DATA_DIR", BASE_DIR / "data"))
MODEL_DIR: Path = Path(os.getenv("JOBS_MODEL_DIR", BASE_DIR / "models"))
LOG_DIR: Path = Path(os.getenv("JOBS_LOG_DIR", BASE_DIR / "logs"))

DB_PATH: Path = DATA_DIR / "jobs.db"
MODEL_PATH: Path = MODEL_DIR / "salary_model.joblib"
METRICS_PATH: Path = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Taxonomies
# --------------------------------------------------------------------------- #
ROLE_CATEGORIES: list[str] = [
    "Data Scientist",
    "Data Engineer",
    "ML Engineer",
    "Software Engineer",
]
EXPERIENCE_LEVELS: list[str] = ["Junior", "Mid", "Senior", "Lead"]
LOCATIONS: list[str] = [
    "San Francisco",
    "New York",
    "Seattle",
    "Austin",
    "Boston",
    "Chicago",
    "Denver",
    "Remote",
]

# --------------------------------------------------------------------------- #
# Skill dictionary: display name -> case-insensitive regex.
# Word boundaries keep "SQL" from matching "NoSQL" and "Java" from "JavaScript".
# --------------------------------------------------------------------------- #
SKILL_PATTERNS: dict[str, str] = {
    "Python": r"\bpython\b",
    "SQL": r"\b(?:sql|postgres(?:ql)?|mysql|t-sql)\b",
    "AWS": r"\b(?:aws|amazon web services|sagemaker|redshift)\b",
    "Azure": r"\bazure\b",
    "GCP": r"\b(?:gcp|google cloud(?: platform)?|bigquery)\b",
    "Docker": r"\bdocker(?:ized|ised)?\b",
    "Kubernetes": r"\b(?:kubernetes|k8s)\b",
    "Spark": r"\b(?:apache spark|pyspark|spark)\b",
    "Airflow": r"\bairflow\b",
    "Snowflake": r"\bsnowflake\b",
    "Kafka": r"\bkafka\b",
    "dbt": r"\bdbt\b",
    "PyTorch": r"\bpytorch\b",
    "TensorFlow": r"\b(?:tensorflow|keras)\b",
    "Scikit-learn": r"\b(?:scikit[- ]?learn|sklearn)\b",
    "LLMs": r"\b(?:llms?|large language models?|generative ai|langchain)\b",
    "Tableau": r"\b(?:tableau|power ?bi)\b",
    "Java": r"\bjava\b",
    "TypeScript": r"\btypescript\b",
}
_COMPILED_SKILLS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in SKILL_PATTERNS.items()
}


def skill_column(skill: str) -> str:
    """Return the dataframe column name for a skill flag (e.g. 'scikit_learn')."""
    return "skill_" + re.sub(r"\W+", "_", skill.lower()).strip("_")


def extract_skill_flags(text: str | None) -> dict[str, int]:
    """Extract binary skill flags from free text using the regex dictionary.

    Args:
        text: A raw or cleaned job description (``None`` is treated as empty).

    Returns:
        Mapping of skill display name to 0/1.
    """
    haystack = text or ""
    return {name: int(bool(rx.search(haystack))) for name, rx in _COMPILED_SKILLS.items()}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOGGING_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: str = "app.log") -> None:
    """Configure console + rotating-file logging exactly once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_DIR / log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only filesystem, etc.
        root.warning("File logging disabled: %s", exc)

    _LOGGING_CONFIGURED = True
