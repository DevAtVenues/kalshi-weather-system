"""
Forward validation — the honest measure of whether the redesigned engine's edge is real.

Grades every logged pick (data/signals/signals_log.jsonl) against the actual settled
NWS high (IEM, the source Kalshi uses), then reports calibration + realized win-rate
sliced by the redesign dimensions: ensemble-vs-rule pricing, score tier, and actionable
picks (did we beat the price?). This is the only honest validation available — no
historical ensemble backtest exists (Open-Meteo keeps members ~3 days), so the edge is
proven forward, from accumulating picks-vs-outcomes, or not at all.

Usage:
    .venv/bin/python scripts/validate_forward.py [--since YYYY-MM-DD] [--min-n N]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.outcome_tracker import grade_signals, forward_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward-validate the live engine's picks")
    ap.add_argument("--since", metavar="YYYY-MM-DD", help="Only grade settlements on/after this date")
    ap.add_argument("--min-n", type=int, default=1, help="Min graded contracts to report (default 1)")
    ap.add_argument("--no-grade", action="store_true", help="Skip grading; report existing outcomes only")
    args = ap.parse_args()

    if not args.no_grade:
        since = date.fromisoformat(args.since) if args.since else None
        n = grade_signals(since=since)
        print(f"(graded {n} new signal rows)")
    forward_report(min_n=args.min_n)


if __name__ == "__main__":
    main()
