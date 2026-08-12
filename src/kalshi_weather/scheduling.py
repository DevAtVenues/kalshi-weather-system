"""Shared once-per-window run guard.

Both a cron entry and a launchd agent schedule the same jobs (run_live,
archive_forecasts). launchd exists to CATCH UP a run missed while the laptop slept
(plain cron doesn't). When the machine is awake, cron and launchd can both fire the
same window — this guard stops the duplicate.

Mechanism: a non-blocking flock (no two instances run at once) + a `last_run` marker
(skip a run that lands within `min_gap_s` of the previous one). Scheduled windows are
hours apart, so a legitimate run never trips the gap; only a cron/launchd double-fire
(seconds–minutes apart) does.

Fails OPEN: any unreadable/missing state → allow the run. A missed data-logging run
is worse than a rare duplicate, so the guard never blocks on doubt.
"""
from __future__ import annotations

import fcntl
import time
from pathlib import Path


def acquire_slot(state_dir: Path, name: str, min_gap_s: float, force: bool = False):
    """Return an open, flock-held file handle if this invocation should run, else None.
    Caller MUST keep the handle alive for the whole run and pass it to mark_and_release."""
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(state_dir / f"{name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None                       # another instance holds it → skip
    if not force:
        marker = state_dir / f"{name}.last_run"
        try:
            if marker.exists() and (time.time() - float(marker.read_text().strip())) < min_gap_s:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
                return None               # ran too recently (cron+launchd double) → skip
        except (ValueError, OSError):
            pass                          # unreadable marker → fail open (run)
    return lock


def mark_and_release(state_dir: Path, name: str, lock) -> None:
    """Stamp the completion time and release the lock. Call after a successful run."""
    try:
        (state_dir / f"{name}.last_run").write_text(str(time.time()))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
