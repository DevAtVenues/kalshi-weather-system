"""
Print a calibration and city performance report from graded signals.

Usage:
    .venv/bin/python scripts/show_performance.py [--min-signals N]

Run grade_signals.py first to populate data/outcomes/signal_outcomes.jsonl.
"""
import argparse

from kalshi_weather.outcome_tracker import calibration_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Show calibration and city performance")
    parser.add_argument(
        "--min-signals",
        type=int,
        default=5,
        metavar="N",
        help="Minimum signals per city to include in city table (default: 5)",
    )
    args = parser.parse_args()
    calibration_report(min_signals=args.min_signals)


if __name__ == "__main__":
    main()
