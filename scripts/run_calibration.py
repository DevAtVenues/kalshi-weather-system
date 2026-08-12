"""
Bias-correction calibration pipeline.

Fetches historical GFS forecasts and NWS CLI labels for all configured
stations, computes per-station per-month bias, and saves the result to
data/calibration/bias.parquet.

The dashboard loads this file at startup to apply real-time corrections.
Re-run weekly (or after adding new stations) to keep calibration fresh.

Run:
  .venv/bin/python scripts/run_calibration.py

Options (env vars):
  CALIB_CITIES  — comma-separated city keys to limit scope (default: all)
  CALIB_YEARS   — comma-separated years (default: 2022,2023,2024)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.calibration.bias import CALIB_YEARS, compute_bias_table
from kalshi_weather.config import load_stations

_CITY_FILTER = os.getenv("CALIB_CITIES", "")
_YEAR_FILTER = os.getenv("CALIB_YEARS", "")

CITY_FILTER = set(_CITY_FILTER.upper().split(",")) if _CITY_FILTER else set()
YEARS = (
    [int(y) for y in _YEAR_FILTER.split(",") if y.strip()]
    if _YEAR_FILTER else CALIB_YEARS
)


def main() -> None:
    stations_cfg = load_stations()
    if CITY_FILTER:
        stations_cfg = {k: v for k, v in stations_cfg.items() if k in CITY_FILTER}

    print("=" * 56)
    print("  Kalshi Weather — Forecast Bias Calibration")
    print(f"  Cities : {', '.join(stations_cfg.keys())}")
    print(f"  Years  : {YEARS}")
    print(f"  Source : Open-Meteo Historical Forecast + IEM CLI")
    print("=" * 56)
    print()

    bias_df = compute_bias_table(stations_cfg, YEARS)

    if bias_df.empty:
        print("\nERROR: No calibration data produced.")
        sys.exit(1)

    print(f"\nCalibration complete — {len(bias_df)} station-month pairs.")
    print("\nPer-station June bias (month 6):")
    june = bias_df[bias_df["month"] == 6].set_index("station")[["bias_f", "std_f", "n"]]
    print(june.sort_values("bias_f").to_string(float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
