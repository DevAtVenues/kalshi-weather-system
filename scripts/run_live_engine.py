"""
Live edge engine — the persistent replacement for the 6-hour run_live cron.

Runs the EXACT same cycle as scripts/run_live.py (run_cycle): ensemble-priced
signals (Move 1) -> evidence-gated forecast picks (push_forecast_picks: evidence-
scaled edge bar + multi-model agreement + calibration-error ceiling) -> intraday
observation locks (push_obs_locks) -> deduped ntfy pushes -> signal log. Only the
CADENCE changes: every MONITOR_PICK_SECONDS (default 600 = 10 min) instead of once
per 6 hours, so an edge is caught the minute it becomes actionable rather than at
the next cron tick. Same gating, same notifications, no second policy. Alert-only.

Run:
  .venv/bin/python scripts/run_live_engine.py
  .venv/bin/python scripts/run_live_engine.py --once               # one cycle, then exit
  MONITOR_PICK_SECONDS=600 .venv/bin/python scripts/run_live_engine.py

Rate budget: one cycle fetches the ensemble per validated city/date plus lazy
multi-model agreement calls for edge-cleared candidates only — comfortably within
Open-Meteo's free tier (10k/day) at a 10-15 min cadence.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))   # import the sibling run_live module

import run_live  # noqa: E402  (provides run_cycle, identical gating to the cron)
from kalshi_weather.live.risk import RiskManager  # noqa: E402
from kalshi_weather.live.runner import CITY_CONFIGS  # noqa: E402

PICK_INTERVAL = int(os.getenv("MONITOR_PICK_SECONDS", "600"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Live edge engine (persistent, event-cadence)")
    ap.add_argument("--once", action="store_true", help="Run one cycle, then exit (for testing)")
    ap.add_argument("--cities", nargs="+", default=None, help="City keys (default: all)")
    ap.add_argument("--interval", type=int, default=PICK_INTERVAL,
                    help=f"Seconds between cycles (default {PICK_INTERVAL})")
    ap.add_argument("--no-log", action="store_true", help="Skip the signal log")
    args = ap.parse_args()

    cities = args.cities or list(CITY_CONFIGS.keys())
    risk = RiskManager()
    risk.reset_if_new_day()

    print("=" * 60)
    print("  Kalshi Live Edge Engine (replaces the 6h cron)")
    print(f"  Cycle interval: {args.interval}s  |  Cities: {cities}")
    print("=" * 60)

    while True:
        t0 = time.monotonic()
        risk.reset_if_new_day()
        try:
            s = run_live.run_cycle(cities, risk, log=not args.no_log)
            print(f"[{datetime.now(timezone.utc):%H:%M UTC}] cycle done — "
                  f"{s['picks']} picks pushed, {s['locks']} obs-locks, "
                  f"{s['filtered']} filtered, {s['n_signals']} contracts scored")
        except Exception as e:
            traceback.print_exc()
            print(f"[engine] cycle error (continuing): {e}")

        if args.once:
            print("\n[--once] single cycle complete.")
            break
        time.sleep(max(5.0, args.interval - (time.monotonic() - t0)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEngine stopped.")
