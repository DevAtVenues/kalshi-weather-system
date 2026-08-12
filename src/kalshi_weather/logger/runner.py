"""
Live logger main loop.

Two independent schedules run in the same process:

  Ensemble (every ENSEMBLE_INTERVAL seconds):
    Checks which GEFS / ECMWF runs should be available by now.
    Fetches any run × station pair not yet stored.
    One parquet file per (model, init_time, station) — idempotent.

  Orderbook (every ORDERBOOK_INTERVAL seconds):
    Snapshots full bid/ask depth for every open weather market.
    Appends to today's parquet file.

Keep this process running continuously. Missed periods cannot be recovered —
the historical ensemble member data and historical orderbook depth are gone
the moment the live window closes.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

from kalshi_weather.tz import UTC
from pathlib import Path

from kalshi_weather import logger as _pkg   # noqa: F401 — ensure package importable
from kalshi_weather.logger import ensemble, orderbook

ENSEMBLE_INTERVAL  = 900   # 15 min — frequent enough to catch new runs quickly
ORDERBOOK_INTERVAL = 300   # 5 min  — fine-grained depth capture

log = logging.getLogger(__name__)


def run(stations_cfg: dict, log_dir: Path) -> None:
    """
    Main blocking loop. Call from the launcher script.
    Ctrl+C to stop cleanly.
    """
    ensemble_dir  = log_dir / "ensemble"
    orderbook_dir = log_dir / "orderbook"

    ensemble_dir.mkdir(parents=True, exist_ok=True)
    orderbook_dir.mkdir(parents=True, exist_ok=True)

    log.info("logger started — %d cities", len(stations_cfg))
    log.info("ensemble  → %s  (every %d min)", ensemble_dir, ENSEMBLE_INTERVAL // 60)
    log.info("orderbook → %s  (every %d min)", orderbook_dir, ORDERBOOK_INTERVAL // 60)

    last_ensemble  = 0.0
    last_orderbook = 0.0

    while True:
        now = time.monotonic()
        utc = datetime.now(UTC).strftime("%H:%M UTC")

        # ── Orderbook snapshot ────────────────────────────────────────
        if now - last_orderbook >= ORDERBOOK_INTERVAL:
            try:
                rows = orderbook.snapshot_all(stations_cfg)
                orderbook.append_snapshot(rows, orderbook_dir)
                log.info("[%s] orderbook: %d levels across all markets", utc, len(rows))
            except Exception as exc:
                log.error("orderbook run failed: %s", exc)
            last_orderbook = now

        # ── Ensemble fetch ────────────────────────────────────────────
        if now - last_ensemble >= ENSEMBLE_INTERVAL:
            try:
                n = ensemble.fetch_missing(stations_cfg, ensemble_dir)
                if n:
                    log.info("[%s] ensemble: fetched %d new run/station pairs", utc, n)
                else:
                    log.debug("[%s] ensemble: nothing new", utc)
            except Exception as exc:
                log.error("ensemble run failed: %s", exc)
            last_ensemble = now

        time.sleep(10)
