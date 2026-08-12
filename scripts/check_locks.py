"""
Intraday OBSERVATION-LOCK checker (lightweight, high-frequency).

Why this exists separately from run_live.py:
  A genuine "lock" — the day's high has been OBSERVED past a contract boundary, so
  the outcome is certain — is only ACTIONABLE in a narrow intraday window: after
  the afternoon peak confirms it, but before the market reprices to 0/1. The main
  run_live.py cron fires only 4×/day and structurally walks in after that window
  has closed (by evening every lock is priced at 1¢). This checker is cheap enough
  to run every ~30 min through the afternoon to catch locks while they can still
  be bought.

It does NOT fetch forecasts or score model edge (that's run_live.py's job) — it
only pulls open markets + the running daily high and reuses run_live.push_obs_locks,
which already gates by the dashboard's 5-25¢ actionable-edge band and dedupes via
the shared notified_today.json state. Run from cron, e.g. every 30 min, 11:00-20:00.

Run:   .venv/bin/python scripts/check_locks.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))            # so `scripts.run_live` resolves under cron

import pandas as pd

from kalshi_weather.live.runner import CITY_CONFIGS, fetch_open_markets
from scripts.run_live import push_obs_locks, NOTIFY_STATE  # reuse lock logic + state path


def main() -> None:
    print("=== Intraday Lock Checker ===")

    today_str = date.today().isoformat()
    if NOTIFY_STATE.exists():
        try:
            state = json.loads(NOTIFY_STATE.read_text())
            if state.get("date") != today_str:
                state = {"date": today_str, "picks": {}}
        except Exception:
            state = {"date": today_str, "picks": {}}
    else:
        state = {"date": today_str, "picks": {}}

    frames: list[pd.DataFrame] = []
    for city_key, cfg in CITY_CONFIGS.items():
        try:
            contracts = fetch_open_markets(cfg["series"], cfg["nws_station"])
        except Exception as exc:
            print(f"  {city_key}: market fetch failed: {exc}")
            continue
        if contracts.empty:
            continue
        contracts = contracts.copy()
        contracts["city"] = city_key
        # push_obs_locks needs a spread column (fetch_open_markets doesn't add one)
        contracts["spread"] = contracts["yes_ask_dollars"] - contracts["yes_bid_dollars"]
        frames.append(contracts)

    if not frames:
        print("  No open markets.")
        return

    signals = pd.concat(frames, ignore_index=True)
    n = push_obs_locks(signals, state)
    if not n:
        print("  No actionable obs-locks right now.")

    NOTIFY_STATE.parent.mkdir(parents=True, exist_ok=True)
    NOTIFY_STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
