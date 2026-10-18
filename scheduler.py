"""Long-running scheduler: keeps data fresh and the model current without manual runs.

Jobs (each isolated so one failure never stops the others):
  * simulated scrape  - every ``--simulated-every`` minutes   (cheap, default 30)
  * Remotive scrape   - every ``--remotive-every`` minutes    (default 360 = 4x/day,
                        the maximum Remotive's terms recommend; responses are also cached)
  * retrain           - every ``--retrain-every`` minutes     (default 1440), only if
                        new rows arrived since the last training

Usage:
    python scheduler.py                       # run forever
    python scheduler.py --no-remotive         # offline / simulator only
    python scheduler.py --run-once            # execute every job once, then exit
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler

from config import DB_PATH, MODEL_PATH, setup_logging
from database import count_jobs
from model import train_and_save
from scraper import run_scrape

logger = logging.getLogger(__name__)

_last_trained_rows = {"simulated": -1}


def safe(name: str, func: Callable[[], None]) -> Callable[[], None]:
    """Wrap a job so exceptions are logged instead of killing the scheduler."""
    def runner() -> None:
        logger.info("Job '%s' started", name)
        try:
            func()
            logger.info("Job '%s' finished", name)
        except Exception:
            logger.exception("Job '%s' failed (will retry on next schedule)", name)
    runner.__name__ = f"job_{name}"
    return runner


def job_simulated(pages: int) -> None:
    run_scrape(pages=pages, source="simulated")


def job_remotive() -> None:
    run_scrape(source="remotive")


def job_retrain(track: bool) -> None:
    rows = count_jobs(DB_PATH)
    if MODEL_PATH.exists() and rows == _last_trained_rows["simulated"]:
        logger.info("Retrain skipped: no new rows since last training (%d)", rows)
        return
    train_and_save(track=track)
    _last_trained_rows["simulated"] = rows


def main() -> None:
    """CLI entry point."""
    setup_logging()
    p = argparse.ArgumentParser(description="Scheduled ingestion + retraining")
    p.add_argument("--simulated-every", type=float, default=30, help="minutes")
    p.add_argument("--remotive-every", type=float, default=360, help="minutes (>= 360 recommended)")
    p.add_argument("--retrain-every", type=float, default=1440, help="minutes")
    p.add_argument("--pages", type=int, default=20)
    p.add_argument("--no-remotive", action="store_true")
    p.add_argument("--track", action="store_true", help="log retrains to MLflow")
    p.add_argument("--run-once", action="store_true")
    args = p.parse_args()

    jobs: list[tuple[str, Callable[[], None], float]] = [
        ("scrape_simulated", lambda: job_simulated(args.pages), args.simulated_every),
        ("retrain", lambda: job_retrain(args.track), args.retrain_every),
    ]
    if not args.no_remotive:
        jobs.insert(1, ("scrape_remotive", job_remotive, args.remotive_every))

    if args.run_once:
        for name, func, _ in jobs:
            safe(name, func)()
        return

    scheduler = BlockingScheduler()
    for name, func, minutes in jobs:
        scheduler.add_job(safe(name, func), "interval", minutes=minutes, id=name,
                          next_run_time=datetime.now(), max_instances=1, coalesce=True)
        logger.info("Scheduled '%s' every %.4g min", name, minutes)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
