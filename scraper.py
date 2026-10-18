"""Resilient ingestion layer: fetch -> parse messy HTML -> normalise -> upsert.

Two interchangeable data sources share one pipeline:

* ``SimulatedFetcher`` - a built-in job-board simulator that renders *messy*
  HTML (three different card layouts, HTML entities, inconsistent salary
  formats, missing fields, transient 503s and captcha pages). It keeps the
  project self-contained, reproducible (``--seed``) and IP-block-proof.
* ``HttpFetcher`` - a polite real-world fetcher (rotating User-Agents, robots.txt
  check, randomised delays). Point it at a site and adapt ``FIELD_SELECTORS``.

Usage:
    python scraper.py --pages 40 --seed 42
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

from config import DATA_DIR, DB_PATH, setup_logging
from database import JobListing, UpsertResult, log_scrape_run, upsert_jobs, utc_now

logger = logging.getLogger(__name__)

# =========================================================================== #
# Text normalisation helpers (pure functions - easy to unit test)
# =========================================================================== #
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_NUM_RE = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(k)?")
_SALARY_IN_TEXT_RE = re.compile(
    r"\$\s?\d{2,3}(?:,\d{3})?\s?k?\s?(?:-|\u2013|\u2014|to)\s?\$?\s?\d{2,3}(?:,\d{3})?\s?k?"
    r"(?:\s?(?:/|per)\s?(?:yr|year|hr|hour))?",
    re.IGNORECASE,
)
_NON_USD_RE = re.compile(r"[\u20ac\u00a3\u20b9]|\b(?:eur|gbp|cad|aud|inr|chf|pln|brl)\b")
MIN_ANNUAL_SALARY = 20_000.0
MAX_ANNUAL_SALARY = 1_000_000.0
HOURS_PER_YEAR = 2080


def clean_text(raw: str) -> str:
    """Decode entities, drop zero-width chars/NBSP and collapse whitespace."""
    text = html.unescape(raw or "")
    text = _ZERO_WIDTH_RE.sub("", text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_company(raw: str) -> str:
    """Tidy a company name; SHOUTING CASE becomes Title Case."""
    name = clean_text(raw).strip(" ,.;-")
    name = re.sub(r"[,\s]+(?:inc|llc|ltd|corp|co)\.?$", "", name, flags=re.IGNORECASE)
    return name.title() if name.isupper() else name


def parse_salary(raw: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Parse a messy salary string into an annualised (min, max) in USD.

    Handles "$150k - $190k", "150,000 - 190,000 USD", "$72-$91 per hour",
    "120-150k" (suffix shared across the range) and junk like "Competitive".

    Returns:
        ``(None, None)`` when no plausible salary can be extracted.
    """
    if not raw:
        return None, None
    text = raw.lower()
    if _NON_USD_RE.search(text):  # never treat EUR/GBP/... as USD
        logger.debug("Non-USD salary %r skipped", raw)
        return None, None
    matches = _NUM_RE.findall(text)
    if not matches:
        return None, None

    values: list[tuple[float, bool]] = []
    for number, k_suffix in matches[:2]:
        value = float(number.replace(",", ""))
        values.append((value * 1000 if k_suffix else value, bool(k_suffix)))

    # "120-150k": the 'k' only appears on the second number.
    if len(values) == 2 and values[1][1] and not values[0][1] and values[0][0] < 1000:
        values[0] = (values[0][0] * 1000, True)

    amounts = [v for v, _ in values]
    if re.search(r"/\s*h(?:ou)?r|per\s+hour|hourly|an\s+hour", text):
        amounts = [a * HOURS_PER_YEAR for a in amounts]
    elif re.search(r"per\s+month|/\s*mo\b|monthly", text):
        amounts = [a * 12 for a in amounts]

    low, high = min(amounts), max(amounts)
    if low < MIN_ANNUAL_SALARY or high > MAX_ANNUAL_SALARY:
        logger.debug("Salary %r outside plausible range, discarding", raw)
        return None, None
    return float(low), float(high)


_LOCATION_RULES: list[tuple[str, str]] = [
    (r"remote|anywhere|work from home|wfh", "Remote"),
    (r"san francisco|\bsf\b|bay area|palo alto|mountain view|san jose", "San Francisco"),
    (r"new york|nyc|\bny\b|brooklyn", "New York"),
    (r"seattle|bellevue|redmond", "Seattle"),
    (r"austin", "Austin"),
    (r"boston|cambridge, ma", "Boston"),
    (r"chicago", "Chicago"),
    (r"denver|boulder", "Denver"),
]


def normalize_location(raw: str) -> str:
    """Map messy location strings ('SF Bay Area', 'NYC (Hybrid)') to a canonical city."""
    text = clean_text(raw).lower()
    for pattern, canonical in _LOCATION_RULES:
        if re.search(pattern, text):
            return canonical
    return "Other"


def categorize_role(title: str) -> str:
    """Map a free-text job title onto the four tracked role families."""
    t = title.lower()
    if re.search(r"\bml\b|machine learning|mlops|\bai\b", t):
        return "ML Engineer"
    if re.search(r"data scien", t):
        return "Data Scientist"
    if re.search(r"\bdata\b.*\bengineer|analytics engineer|\betl\b", t):
        return "Data Engineer"
    if re.search(r"software|developer|backend|back-end|full[- ]?stack|\bsde\b", t):
        return "Software Engineer"
    return "Other"


def normalize_experience(title: str, description: str) -> str:
    """Infer Junior/Mid/Senior/Lead from the title, falling back to 'N+ years'."""
    t = title.lower()
    if re.search(r"\b(lead|staff|principal|head|director)\b", t):
        return "Lead"
    if re.search(r"\b(senior|sr)\b", t) or re.search(r"\b(iii|iv)\b", t):
        return "Senior"
    if re.search(r"\b(junior|jr|entry[- ]level|associate|intern|graduate)\b", t):
        return "Junior"
    if re.search(r"\bmid[- ]?level\b|\bii\b", t):
        return "Mid"
    match = re.search(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", description.lower())
    if match:
        years = int(match.group(1))
        if years <= 2:
            return "Junior"
        if years <= 4:
            return "Mid"
        return "Senior" if years <= 8 else "Lead"
    return "Mid"


# =========================================================================== #
# Fetching layer
# =========================================================================== #
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Edg/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]


class FetchError(Exception):
    """Raised when a page cannot be retrieved (network error, 5xx, blocked...)."""


class RateLimiter:
    """Randomised politeness delay between consecutive requests."""

    def __init__(self, min_delay: float, max_delay: float) -> None:
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError("Require 0 <= min_delay <= max_delay")
        self._min, self._max = min_delay, max_delay
        self._last_request: Optional[float] = None

    def wait(self) -> None:
        """Sleep so that at least a random delay has elapsed since the last call."""
        if self._last_request is not None:
            target = random.uniform(self._min, self._max)
            remaining = target - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()


class BaseFetcher(ABC):
    """Interface every data source implements."""

    name: str = "base"

    @abstractmethod
    def fetch_page(self, page: int) -> str:
        """Return raw HTML for the given 1-based page number."""


class HttpFetcher(BaseFetcher):
    """Polite HTTP fetcher with rotating User-Agents and robots.txt compliance."""

    name = "http"

    def __init__(self, url_template: str, timeout: float = 10.0) -> None:
        if "{page}" not in url_template:
            raise ValueError("url_template must contain a '{page}' placeholder")
        self.url_template = url_template
        self.timeout = timeout
        self._session = requests.Session()
        self._robots: Optional[robotparser.RobotFileParser] = None

    def _allowed_by_robots(self, url: str) -> bool:
        if self._robots is None:
            parts = urlparse(url)
            parser = robotparser.RobotFileParser(f"{parts.scheme}://{parts.netloc}/robots.txt")
            try:
                parser.read()
            except Exception as exc:  # network failure: be conservative
                logger.error("Could not read robots.txt (%s); refusing to crawl", exc)
                return False
            self._robots = parser
        return self._robots.can_fetch("*", url)

    def fetch_page(self, page: int) -> str:
        url = self.url_template.format(page=page)
        if not self._allowed_by_robots(url):
            raise FetchError(f"robots.txt disallows {url}")
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = self._session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FetchError(f"GET {url} failed: {exc}") from exc
        return response.text


# --------------------------------------------------------------------------- #
# Built-in job-board simulator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoleProfile:
    """Generative profile for one role family."""

    base_salary: int
    titles: list[str]
    responsibilities: list[str]
    skill_probs: dict[str, float]


ROLE_PROFILES: dict[str, RoleProfile] = {
    "Data Scientist": RoleProfile(
        125_000,
        ["Data Scientist", "Data Scientist, Product Analytics", "Applied Data Scientist"],
        [
            "Translate ambiguous business questions into measurable analyses",
            "Design and evaluate experiments and A/B tests",
            "Build predictive models and communicate findings to stakeholders",
            "Partner with product and engineering to ship data-driven features",
        ],
        {"Python": .95, "SQL": .85, "Scikit-learn": .6, "PyTorch": .3, "TensorFlow": .2,
         "AWS": .3, "Azure": .1, "GCP": .1, "Docker": .15, "Tableau": .3, "LLMs": .25, "Spark": .15},
    ),
    "Data Engineer": RoleProfile(
        130_000,
        ["Data Engineer", "Analytics Engineer", "Data Platform Engineer"],
        [
            "Design, build and maintain scalable data pipelines",
            "Model and curate datasets for analytics and machine learning teams",
            "Monitor data quality and own pipeline reliability",
            "Optimize performance and cost of our data platform",
        ],
        {"Python": .8, "SQL": .95, "Spark": .6, "Airflow": .55, "Snowflake": .4, "Kafka": .3,
         "dbt": .35, "AWS": .5, "Azure": .15, "GCP": .2, "Docker": .35, "Kubernetes": .2, "Java": .1},
    ),
    "ML Engineer": RoleProfile(
        145_000,
        ["Machine Learning Engineer", "ML Engineer", "AI/ML Engineer", "MLOps Engineer"],
        [
            "Productionize machine learning models and serve them at scale",
            "Build training and evaluation pipelines",
            "Monitor model performance and drift in production",
            "Collaborate with researchers to turn prototypes into products",
        ],
        {"Python": .97, "PyTorch": .6, "TensorFlow": .3, "LLMs": .45, "Docker": .6, "Kubernetes": .4,
         "AWS": .5, "GCP": .2, "Azure": .1, "SQL": .5, "Spark": .2, "Scikit-learn": .4, "Airflow": .15},
    ),
    "Software Engineer": RoleProfile(
        120_000,
        ["Software Engineer", "Backend Software Engineer", "Full-Stack Developer",
         "Software Development Engineer"],
        [
            "Design and build reliable backend services and APIs",
            "Write clean, well-tested code and participate in code reviews",
            "Contribute to system design and architecture decisions",
            "Improve observability and developer tooling",
        ],
        {"Python": .4, "Java": .45, "TypeScript": .35, "SQL": .5, "AWS": .5, "Docker": .5,
         "Kubernetes": .3, "Kafka": .15, "Azure": .1, "GCP": .15},
    ),
}

LEVEL_MULTIPLIER = {"Junior": 0.72, "Mid": 1.0, "Senior": 1.27, "Lead": 1.52}
LEVEL_WEIGHTS = {"Junior": 0.2, "Mid": 0.35, "Senior": 0.3, "Lead": 0.15}
LEVEL_YEARS = {"Junior": (0, 2), "Mid": (3, 4), "Senior": (5, 8), "Lead": (9, 13)}
LEVEL_TITLE_FORMATS = {
    "Junior": ["Junior {t}", "Jr. {t}", "Associate {t}", "Entry-Level {t}"],
    "Mid": ["{t}", "{t} II", "{t}", "Mid-Level {t}"],
    "Senior": ["Senior {t}", "Sr. {t}", "{t} III", "Senior {t}"],
    "Lead": ["Lead {t}", "Staff {t}", "Principal {t}"],
}
LOCATION_MULTIPLIER = {
    "San Francisco": 1.18, "New York": 1.12, "Seattle": 1.10, "Boston": 1.03,
    "Austin": 0.98, "Remote": 0.97, "Denver": 0.96, "Chicago": 0.94,
}
LOCATION_WEIGHTS = {
    "San Francisco": .18, "New York": .17, "Seattle": .12, "Austin": .09,
    "Boston": .08, "Chicago": .07, "Denver": .05, "Remote": .24,
}
LOCATION_VARIANTS = {
    "San Francisco": ["San Francisco, CA", "SF Bay Area", "San Francisco (Hybrid)", "Palo Alto, CA"],
    "New York": ["New York, NY", "NYC", "New York City (Hybrid)", "Brooklyn, NY"],
    "Seattle": ["Seattle, WA", "Bellevue, WA", "Seattle WA - Onsite"],
    "Austin": ["Austin, TX", "Austin TX (Hybrid)"],
    "Boston": ["Boston, MA", "Cambridge, MA"],
    "Chicago": ["Chicago, IL", "Chicago (Onsite)"],
    "Denver": ["Denver, CO", "Boulder, CO"],
    "Remote": ["Remote", "Remote - US", "Remote (United States)", "Anywhere (Remote)"],
}
# Annual USD premium that each skill adds (the "ground truth" the model must learn).
SKILL_PREMIUM = {
    "LLMs": 16_000, "PyTorch": 9_000, "Kubernetes": 6_000, "Spark": 6_000, "Kafka": 6_000,
    "TensorFlow": 4_000, "Snowflake": 4_000, "Airflow": 3_000, "AWS": 4_000, "GCP": 3_000,
    "Azure": 2_000, "Docker": 2_500, "dbt": 2_500, "Python": 2_000, "Scikit-learn": 1_500,
    "TypeScript": 1_000, "Java": 1_000, "Tableau": 0, "SQL": 0,
}
SKILL_PHRASES = {
    "AWS": ["AWS", "Amazon Web Services", "AWS (S3, EC2)"],
    "GCP": ["GCP", "Google Cloud Platform", "BigQuery"],
    "Spark": ["Spark", "PySpark", "Apache Spark"],
    "Kubernetes": ["Kubernetes", "K8s"],
    "LLMs": ["LLMs", "large language models", "Generative AI", "LangChain"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Python": ["Python", "Python 3", "python"],
    "SQL": ["SQL", "PostgreSQL", "SQL"],
    "TensorFlow": ["TensorFlow", "Keras"],
    "Tableau": ["Tableau", "Power BI"],
}
COMPANIES = [
    "Northwind Analytics", "Lumen Labs", "Orbital Dynamics", "BlueHarbor Systems", "Quanta Retail",
    "Helix Health", "Redwood Payments", "Vertex Mobility", "Silverline Media", "Atlas Robotics",
    "Pioneer Energy", "Nimbus Cloud", "Ironclad Security", "Cobalt Genomics", "Brightpath Learning",
    "Summit Logistics", "Evergreen Foods", "Zephyr Travel", "Granite Insurance", "Polaris Gaming",
    "Meridian Bank", "Tidal Commerce", "Kestrel Aerospace", "Foxglove Studios",
]
INTRO_TEMPLATES = [
    "{company} is hiring a {title} to join our growing team.",
    "Join {company} as a {title} and help shape the future of our platform.",
    "We're looking for a {title} to work on high-impact problems at {company}.",
]
BOILERPLATE = [
    "{company} is an equal opportunity employer.",
    "We offer competitive benefits, flexible time off and a learning stipend.",
    "Our team uses Slack, Jira and Notion to stay in sync.",
    "Apply today \u2014 we review applications on a rolling basis.",
]
CAPTCHA_HTML = "<html><body><h1>Please verify you are human</h1><div class='cf-captcha'></div></body></html>"


def _join_skills(skills: list[str]) -> str:
    return skills[0] if len(skills) == 1 else ", ".join(skills[:-1]) + " and " + skills[-1]


class JobBoardSimulator:
    """Deterministically renders messy job-board HTML for a given (seed, page)."""

    def __init__(self, seed: int, jobs_per_page: int = 25) -> None:
        self.seed = seed
        self.jobs_per_page = jobs_per_page

    # -- ground-truth salary ------------------------------------------------ #
    @staticmethod
    def _true_salary(rng: random.Random, role: str, level: str, location: str, skills: list[str]) -> float:
        loc_mult = LOCATION_MULTIPLIER[location]
        base = ROLE_PROFILES[role].base_salary * LEVEL_MULTIPLIER[level] * loc_mult
        premium = sum(SKILL_PREMIUM.get(s, 0) for s in skills) * loc_mult
        return (base + premium) * rng.gauss(1.0, 0.05)

    @staticmethod
    def _format_salary(rng: random.Random, mid: float) -> str:
        spread = rng.uniform(0.06, 0.14)
        low = round(mid * (1 - spread) / 5000) * 5000
        high = round(mid * (1 + spread) / 5000) * 5000
        style = rng.randrange(6)
        if style == 0:
            return f"${low // 1000}k - ${high // 1000}k"
        if style == 1:
            return f"${low:,} - ${high:,} per year"
        if style == 2:
            return f"{low:,} \u2013 {high:,} USD"
        if style == 3:
            return f"USD {low // 1000}K-{high // 1000}K / yr"
        if style == 4:
            return f"${low / HOURS_PER_YEAR:.0f} - ${high / HOURS_PER_YEAR:.0f} per hour"
        return f"${low // 1000}-{high // 1000}k"

    # -- description -------------------------------------------------------- #
    def _description_html(self, rng: random.Random, company: str, title: str, role: str,
                          years: int, skills: list[str], comp_line: str) -> str:
        phrases = [rng.choice(SKILL_PHRASES.get(s, [s])) for s in skills]
        rng.shuffle(phrases)
        split = max(1, len(phrases) - rng.randint(0, min(2, len(phrases) - 1)))
        required, nice = phrases[:split], phrases[split:]
        skill_sentence = rng.choice([
            "Hands-on experience with {s}.", "Proficiency in {s} is required.", "Our stack: {s}.",
        ]).format(s=_join_skills(required))

        duties = "".join(f"<li>{r}</li>" for r in rng.sample(ROLE_PROFILES[role].responsibilities, 3))
        reqs = [f"{years}+ years of professional experience", skill_sentence]
        if nice:
            reqs.append(f"Nice to have: {_join_skills(nice)}.")
        reqs_html = "".join(f"<li>{r}</li>" for r in reqs)
        boiler = " ".join(rng.sample(BOILERPLATE, 2)).format(company=company)
        comp_html = f"<p><i>{comp_line}</i></p>" if comp_line else ""

        body = (
            f"{rng.choice(INTRO_TEMPLATES).format(company=company, title=title)}<br><br>"
            f"<b>What you'll do</b><ul>{duties}</ul>"
            f"<b>What you'll bring</b><ul>{reqs_html}</ul>{comp_html}{boiler}"
        )
        if rng.random() < 0.05:
            body = body.upper()
        if rng.random() < 0.15:  # sloppy markup: non-breaking and doubled spaces
            body = body.replace(" and ", "&nbsp;and&nbsp; ")
        return body

    # -- single listing ----------------------------------------------------- #
    def _make_listing(self, rng: random.Random) -> dict[str, str]:
        role = rng.choice(list(ROLE_PROFILES))
        profile = ROLE_PROFILES[role]
        level = rng.choices(list(LEVEL_WEIGHTS), weights=list(LEVEL_WEIGHTS.values()))[0]
        location = rng.choices(list(LOCATION_WEIGHTS), weights=list(LOCATION_WEIGHTS.values()))[0]

        skills = [s for s, p in profile.skill_probs.items() if rng.random() < p]
        while len(skills) < 2:
            skills.append(rng.choice(list(profile.skill_probs)))
            skills = list(dict.fromkeys(skills))

        title = rng.choice(LEVEL_TITLE_FORMATS[level]).format(t=rng.choice(profile.titles))
        name = rng.choice(COMPANIES)
        company = rng.choice([name, f"{name} Inc.", f"{name}, Inc", name.upper(), f"  {name}  ", f"{name} LLC"])
        years = rng.randint(*LEVEL_YEARS[level])

        mid = self._true_salary(rng, role, level, location, skills)
        roll = rng.random()
        pay_text, comp_line = "", ""
        if roll < 0.62:                      # salary in its own element
            pay_text = self._format_salary(rng, mid)
        elif roll < 0.72:                    # salary only buried in the description
            low = round(mid * 0.92 / 5000) * 5
            high = round(mid * 1.08 / 5000) * 5
            comp_line = f"Compensation: ${low}k - ${high}k"
        else:                                # undisclosed
            pay_text = rng.choice(["Competitive salary", "DOE", "Not disclosed", "", "Competitive + equity"])

        return {
            "title": title, "company": company,
            "location": rng.choice(LOCATION_VARIANTS[location]),
            "pay": pay_text,
            "description": self._description_html(rng, name, title, role, years, skills, comp_line),
        }

    # -- page rendering ----------------------------------------------------- #
    @staticmethod
    def _render_card(rng: random.Random, job: dict[str, str], idx: int) -> str:
        template = rng.randrange(3)
        title, company, loc = html.escape(job["title"]), html.escape(job["company"]), job["location"]
        pay, desc = html.escape(job["pay"]), job["description"]
        company_html = company if rng.random() > 0.03 else ""   # occasionally missing -> must be dropped
        if template == 0:
            pay_html = f'<span class="pay">{pay}</span>' if pay else ""
            return (
                f'<div class="job-card" data-id="{idx}"><h2 class="job-title">{title}</h2>'
                f'<span class="company">{company_html}</span> <span class="loc">{loc}</span>'
                f'{pay_html}<div class="desc">{desc}</div></div>'
            )
        if template == 1:
            pay_html = f'<div class="comp">Salary: {pay}</div>' if pay else ""
            return (
                f'<li class="result-row"><a class="title-link" href="/job/{idx}">{title}</a>'
                f'<div class="meta"><b>{company_html}</b> &bull; <i>{loc}</i></div>'
                f'<p class="snippet">{desc}</p>{pay_html}</li>'
            )
        pay_html = f'<span class="salary">{pay}</span>' if pay else ""
        return (
            f'<article class="posting"><h3 class="posting-title">{title}</h3>'
            f'<span class="org">{company_html}</span><span class="where">{loc}</span>'
            f'<section class="body">{desc}</section>{pay_html}</article>'
        )

    def render_page(self, page: int) -> str:
        """Render one results page (deterministic for a given seed/page)."""
        rng = random.Random(f"{self.seed}:{page}")
        cards = [self._render_card(rng, self._make_listing(rng), page * 1000 + i)
                 for i in range(self.jobs_per_page)]
        return (
            "<html><head><title>TechJobs - Results</title>"
            "<style>.job-card{margin:4px}</style><script>var _trk={page:%d};</script></head>"
            "<body><nav>Home | Jobs | Salaries</nav><div class='ad'>Sponsored: Learn to code $99/mo</div>"
            "%s<footer>&copy; TechJobs</footer></body></html>" % (page, "\n".join(cards))
        )


class SimulatedFetcher(BaseFetcher):
    """Serves simulator pages and injects transient failures to exercise retries."""

    name = "simulated"

    def __init__(self, seed: int, failure_rate: float = 0.08, captcha_rate: float = 0.03,
                 jobs_per_page: int = 25) -> None:
        self._sim = JobBoardSimulator(seed, jobs_per_page)
        self._failure_rate, self._captcha_rate = failure_rate, captcha_rate
        self._chaos = random.Random()

    def fetch_page(self, page: int) -> str:
        roll = self._chaos.random()
        if roll < self._failure_rate:
            raise FetchError("Simulated HTTP 503 Service Unavailable")
        if roll < self._failure_rate + self._captcha_rate:
            return CAPTCHA_HTML
        return self._sim.render_page(page)


# =========================================================================== #
# Real-world source: Remotive public JSON API
# --------------------------------------------------------------------------- #
# Terms (https://github.com/remotive-com/remote-jobs-api): max ~4 fetches/day,
# >2 requests/min is blocked, and listings must link back to Remotive and name it
# as the source. We therefore (1) issue ONE request per run, (2) cache the raw
# response on disk for >= 6 h, and (3) store each posting's canonical URL so the
# dashboard can attribute it. Listings are never forwarded to third-party boards.
# =========================================================================== #
REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
REMOTIVE_CACHE_TTL_S = 6 * 3600


def parse_remotive_job(job: dict[str, Any]) -> Optional[JobListing]:
    """Convert one Remotive API job object into a :class:`JobListing`.

    Returns ``None`` for irrelevant roles (outside the four tracked families)
    or records missing mandatory fields. Remote-only board, so location is 'Remote'.
    """
    title = clean_text(str(job.get("title") or ""))
    company = clean_company(str(job.get("company_name") or ""))
    if not title or not company:
        return None
    role = categorize_role(title)
    if role == "Other":
        return None

    description = clean_text(
        BeautifulSoup(str(job.get("description") or ""), "html.parser").get_text(" ", strip=True)
    )
    salary_raw = clean_text(str(job.get("salary") or "")) or None
    low, high = parse_salary(salary_raw)
    return JobListing(
        title=title, company=company, location="Remote", description=description,
        experience_level=normalize_experience(title, description), role_category=role,
        salary_raw=salary_raw, salary_min=low, salary_max=high, source="remotive",
        url=str(job.get("url") or "") or None,
    )


class RemotiveSource:
    """Polite client for the Remotive API with an on-disk TTL cache."""

    name = "remotive"

    def __init__(self, api_url: str = REMOTIVE_API_URL, cache_dir: Path = DATA_DIR / "cache",
                 cache_ttl_s: int = REMOTIVE_CACHE_TTL_S, timeout: float = 30.0) -> None:
        self.api_url, self.cache_ttl_s, self.timeout = api_url, cache_ttl_s, timeout
        self.cache_file = Path(cache_dir) / "remotive.json"

    def _read_cache(self) -> Optional[dict[str, Any]]:
        try:
            if time.time() - self.cache_file.stat().st_mtime < self.cache_ttl_s:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return None

    def fetch_payload(self, limiter: RateLimiter, max_retries: int = 3) -> Optional[dict[str, Any]]:
        """Return the API payload from cache or network (``None`` on failure)."""
        cached = self._read_cache()
        if cached is not None:
            logger.info("Remotive: using cached response (TTL %ds) - no network request made", self.cache_ttl_s)
            return cached
        headers = {"User-Agent": random.choice(USER_AGENTS), "Accept": "application/json"}
        for attempt in range(1, max_retries + 1):
            limiter.wait()
            try:
                response = requests.get(self.api_url, headers=headers, timeout=self.timeout)
                if response.status_code == 429:
                    raise FetchError("HTTP 429 rate limited - honouring Remotive's limits")
                response.raise_for_status()
                payload = response.json()
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.cache_file.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except (requests.RequestException, ValueError, FetchError) as exc:
                logger.warning("Remotive attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(2.0 * 2 ** (attempt - 1))
        return None

    def fetch_listings(self, limiter: RateLimiter) -> tuple[list[JobListing], int, bool]:
        """Return ``(listings, dropped, ok)``."""
        payload = self.fetch_payload(limiter)
        if payload is None or not isinstance(payload.get("jobs"), list):
            return [], 0, False
        listings: list[JobListing] = []
        for job in payload["jobs"]:
            try:
                parsed = parse_remotive_job(job)
            except Exception:
                logger.exception("Unexpected error parsing a Remotive job")
                parsed = None
            if parsed is not None:
                listings.append(parsed)
        dropped = len(payload["jobs"]) - len(listings)
        logger.info("Remotive: %d relevant listings kept, %d dropped (other roles / malformed)",
                    len(listings), dropped)
        return listings, dropped, True


# =========================================================================== #
# Parsing layer
# =========================================================================== #
CARD_SELECTORS = ["div.job-card", "li.result-row", "article.posting"]
# Selector fallbacks per field. To scrape a real board, add its selectors here.
FIELD_SELECTORS: dict[str, list[str]] = {
    "title": ["h2.job-title", "a.title-link", "h3.posting-title"],
    "company": ["span.company", "div.meta b", "span.org"],
    "location": ["span.loc", "div.meta i", "span.where"],
    "description": ["div.desc", "p.snippet", "section.body"],
    "salary": ["span.pay", "div.comp", "span.salary"],
}


class JobParser:
    """Turns raw HTML into normalised :class:`JobListing` objects."""

    def __init__(self, source: str) -> None:
        self.source = source

    @staticmethod
    def _first_text(card: Tag, field: str) -> str:
        for selector in FIELD_SELECTORS[field]:
            element = card.select_one(selector)
            if element is not None:
                text = clean_text(element.get_text(" ", strip=True))
                if text:
                    return text
        return ""

    def parse_card(self, card: Tag) -> Optional[JobListing]:
        """Parse one card; return ``None`` if mandatory fields are missing."""
        title = self._first_text(card, "title")
        company = clean_company(self._first_text(card, "company"))
        if not title or not company:
            logger.warning("Dropping malformed card (title=%r, company=%r)", title, company)
            return None

        description = self._first_text(card, "description")
        salary_raw: Optional[str] = self._first_text(card, "salary") or None
        salary_raw = re.sub(r"^\s*salary:\s*", "", salary_raw or "", flags=re.I) or None
        low, high = parse_salary(salary_raw)
        if low is None:  # fall back to scanning the description for "$140k-$170k"
            found = _SALARY_IN_TEXT_RE.search(description)
            if found:
                salary_raw = found.group(0)
                low, high = parse_salary(salary_raw)

        return JobListing(
            title=title,
            company=company,
            location=normalize_location(self._first_text(card, "location")),
            description=description,
            experience_level=normalize_experience(title, description),
            role_category=categorize_role(title),
            salary_raw=salary_raw,
            salary_min=low,
            salary_max=high,
            source=self.source,
        )

    def parse_page(self, page_html: str) -> tuple[list[JobListing], int, int]:
        """Parse a full page.

        Returns:
            ``(listings, dropped_count, cards_found)``
        """
        soup = BeautifulSoup(page_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        cards = soup.select(", ".join(CARD_SELECTORS))
        listings: list[JobListing] = []
        dropped = 0
        for card in cards:
            try:
                parsed = self.parse_card(card)
            except Exception:  # one bad card must never kill the page
                logger.exception("Unexpected error while parsing a card")
                parsed = None
            if parsed is None:
                dropped += 1
            else:
                listings.append(parsed)
        return listings, dropped, len(cards)


# =========================================================================== #
# Pipeline orchestration
# =========================================================================== #
@dataclass(frozen=True)
class ScrapeSummary:
    """Outcome of a scrape run."""

    pages_requested: int
    pages_ok: int
    pages_failed: int
    listings_parsed: int
    listings_dropped: int
    inserted: int
    updated: int
    duration_s: float


def fetch_with_retry(fetcher: BaseFetcher, page: int, limiter: RateLimiter,
                     max_retries: int = 3, backoff: float = 0.5) -> Optional[str]:
    """Fetch a page with rate limiting and exponential backoff; ``None`` on failure."""
    for attempt in range(1, max_retries + 1):
        limiter.wait()
        try:
            return fetcher.fetch_page(page)
        except FetchError as exc:
            logger.warning("Page %d attempt %d/%d failed: %s", page, attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(backoff * 2 ** (attempt - 1) + random.uniform(0, 0.1))
    logger.error("Giving up on page %d after %d attempts", page, max_retries)
    return None


def run_scrape(pages: int = 20, source: str = "simulated", seed: Optional[int] = None,
               db_path=DB_PATH, url_template: Optional[str] = None,
               min_delay: Optional[float] = None, max_delay: Optional[float] = None,
               failure_rate: float = 0.08, remotive_url: str = REMOTIVE_API_URL,
               cache_dir: Path = DATA_DIR / "cache") -> ScrapeSummary:
    """Run the full ingestion pipeline and persist results.

    Args:
        pages: Pages to request (ignored for ``remotive``, which is a single request).
        source: ``"simulated"``, ``"http"`` (generic HTML site) or ``"remotive"`` (real JSON API).
        seed: Simulator seed. ``None`` yields a fresh market snapshot each run.
        db_path: SQLite file to write to.
        url_template: Required for ``source="http"``; must contain ``{page}``.
        min_delay: Lower politeness delay in seconds (source-specific default).
        max_delay: Upper politeness delay in seconds (source-specific default).
        failure_rate: Probability of an injected transient failure (simulator only).
        remotive_url: Override of the Remotive endpoint (used by tests).
        cache_dir: Where the Remotive response cache lives.
    """
    started_at, t0 = utc_now(), time.monotonic()
    all_listings: list[JobListing] = []
    pages_ok = pages_failed = dropped_total = 0

    if source == "remotive":
        limiter = RateLimiter(2.0 if min_delay is None else min_delay, 4.0 if max_delay is None else max_delay)
        listings, dropped_total, ok = RemotiveSource(remotive_url, cache_dir).fetch_listings(limiter)
        all_listings, pages, source_name = listings, 1, "remotive"
        pages_ok, pages_failed = int(ok), int(not ok)
    else:
        if source == "http":
            if not url_template:
                raise ValueError("--url-template is required for source 'http'")
            fetcher: BaseFetcher = HttpFetcher(url_template)
            min_delay = 1.0 if min_delay is None else min_delay
            max_delay = 3.0 if max_delay is None else max_delay
        elif source == "simulated":
            seed = int(time.time()) if seed is None else seed
            fetcher = SimulatedFetcher(seed, failure_rate=failure_rate)
            min_delay = 0.0 if min_delay is None else min_delay
            max_delay = 0.05 if max_delay is None else max_delay
            logger.info("Simulator seed=%d", seed)
        else:
            raise ValueError(f"Unknown source: {source!r}")

        limiter, parser = RateLimiter(min_delay, max_delay), JobParser(fetcher.name)
        source_name = fetcher.name
        for page in range(1, pages + 1):
            page_html = fetch_with_retry(fetcher, page, limiter)
            if page_html is None:
                pages_failed += 1
                continue
            listings, dropped, cards = parser.parse_page(page_html)
            if cards == 0:
                logger.warning("Page %d contained no job cards (captcha / layout change?)", page)
                pages_failed += 1
                continue
            pages_ok += 1
            dropped_total += dropped
            all_listings.extend(listings)
            logger.info("Page %d/%d: %d parsed, %d dropped", page, pages, len(listings), dropped)

    result: UpsertResult = upsert_jobs(all_listings, db_path)
    log_scrape_run(started_at, source_name, pages, pages_ok, pages_failed,
                   len(all_listings), dropped_total, result, db_path)
    summary = ScrapeSummary(pages, pages_ok, pages_failed, len(all_listings), dropped_total,
                            result.inserted, result.updated, time.monotonic() - t0)
    logger.info("Scrape finished: %s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Tech job-market scraper")
    parser.add_argument("--pages", type=int, default=40, help="pages to fetch (25 listings each)")
    parser.add_argument("--source", choices=["simulated", "http", "remotive"], default="simulated",
                        help="remotive = real remote-job listings (max ~4 runs/day, see terms)")
    parser.add_argument("--seed", type=int, default=None, help="simulator seed (default: random)")
    parser.add_argument("--url-template", default=None, help="e.g. https://site/jobs?page={page}")
    parser.add_argument("--min-delay", type=float, default=None)
    parser.add_argument("--max-delay", type=float, default=None)
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()
    run_scrape(args.pages, args.source, args.seed, args.db, args.url_template,
               args.min_delay, args.max_delay)


if __name__ == "__main__":
    main()
