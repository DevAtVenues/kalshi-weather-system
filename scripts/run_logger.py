"""
Live data logger — runs independently of the dashboard.

Logs two proprietary datasets that public APIs do not retain:
  1. GEFS + ECMWF ensemble members at each model run  (every 6h / 12h)
  2. Kalshi orderbook depth snapshots                  (every 5 min)

Keep this running 24/7. Data lost from missed periods cannot be recovered.

Run:
  .venv/bin/python scripts/run_logger.py

Options (env vars):
  LOGGER_DATA_DIR  — where to write parquet files (default: data/logger)
  LOGGER_CITIES    — comma-separated city keys to restrict logging (default: all)
  LOG_LEVEL        — DEBUG / INFO / WARNING (default: INFO)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.config import load_stations
from kalshi_weather.logger.runner import run

DATA_DIR  = Path(os.getenv("LOGGER_DATA_DIR", "data/logger"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_CITY_FILTER = os.getenv("LOGGER_CITIES", "")
CITY_FILTER  = set(_CITY_FILTER.upper().split(",")) if _CITY_FILTER else set()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    stations_cfg = load_stations()
    if CITY_FILTER:
        stations_cfg = {k: v for k, v in stations_cfg.items() if k in CITY_FILTER}

    print("=" * 56)
    print("  Kalshi Weather Live Logger")
    print(f"  Cities   : {', '.join(stations_cfg.keys())}")
    print(f"  Data dir : {DATA_DIR.resolve()}")
    print(f"  Models   : GEFS (31 members)  +  ECMWF ENS (51 members)")
    print(f"  Orderbook: every 5 min")
    print(f"  Ensemble : every 15 min (detects new runs automatically)")
    print("=" * 56)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        run(stations_cfg, DATA_DIR)
    except KeyboardInterrupt:
        print("\nLogger stopped.")


if __name__ == "__main__":
    main()
