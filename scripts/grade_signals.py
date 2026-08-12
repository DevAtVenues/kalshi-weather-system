"""
Grade past signals and open trades against actual settlement temperatures.

Usage:
    .venv/bin/python scripts/grade_signals.py [--since YYYY-MM-DD]

Reads:  data/signals/signals_log.jsonl
        data/trades.json
Writes: data/outcomes/signal_outcomes.jsonl  (appends new records)
        data/trades.json                      (updates outcomes in-place)
"""
import argparse
from datetime import date

from kalshi_weather.outcome_tracker import grade_signals, grade_open_trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade signals and open trades")
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Only grade signals with settlement date on or after this date",
    )
    args = parser.parse_args()

    since: date | None = None
    if args.since:
        since = date.fromisoformat(args.since)

    print("─" * 55)
    print("  GRADING SIGNALS")
    print("─" * 55)
    n_signals = grade_signals(since=since)

    print()
    print("─" * 55)
    print("  GRADING OPEN TRADES")
    print("─" * 55)
    n_trades = grade_open_trades()

    print()
    print(f"Done. Graded {n_signals} signal(s) and {n_trades} trade(s).")


if __name__ == "__main__":
    main()
