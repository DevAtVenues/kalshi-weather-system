"""
Live METAR lock monitor for Kalshi temperature contracts.

Polls METAR every POLL_INTERVAL seconds across all configured cities.
Fires an alert (console + system sound) when a lock condition is met:

  YES_locked   — temperature has already exceeded threshold → contract
                 WILL settle YES.  Consider buying YES.
  NO_locked    — temp exceeded cap on a "less" contract → contract
                 WILL settle NO.   Consider buying NO.
  LOW_confirmed — daily low has reversed 2°F upward → low is locked.

Alert-only: no orders are placed automatically.  CLAUDE.md requires the
risk layer to be complete before any live execution.

Run:
  .venv/bin/python scripts/run_live_monitor.py

Options (env vars):
  MONITOR_POLL_SECONDS   — poll interval (default: 600 = 10 min)
  MONITOR_REFRESH_SECONDS — market refresh interval (default: 1800 = 30 min)
  MONITOR_CITIES         — comma-separated city keys to watch (default: all)
                           e.g.  MONITOR_CITIES=NYC,CHI,MIA
  MONITOR_SOUND          — "1" to play macOS chime on lock (default: 1)
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "src"))

import requests

from kalshi_weather.config import load_stations
from kalshi_weather.ingest.kalshi import _get, _headers
from kalshi_weather.ingest.metar import fetch_metar
from kalshi_weather.monitor.lock import DailyTracker, LockEvent

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL    = int(os.getenv("MONITOR_POLL_SECONDS",   "600"))
REFRESH_INTERVAL = int(os.getenv("MONITOR_REFRESH_SECONDS","1800"))
SOUND_ENABLED    = os.getenv("MONITOR_SOUND", "1") == "1"

_CITY_FILTER = os.getenv("MONITOR_CITIES", "")
CITY_FILTER  = set(_CITY_FILTER.upper().split(",")) if _CITY_FILTER else set()


# ── Market discovery ──────────────────────────────────────────────────────────

def fetch_active_contracts(series_ticker: str, city: str) -> list[dict]:
    """Return all open T-contracts for a series, annotated with city name."""
    try:
        data = _get("/markets", {
            "series_ticker": series_ticker,
            "status":        "open",
            "limit":         200,
        })
    except requests.HTTPError as e:
        print(f"  [warn] {series_ticker}: HTTP {e}")
        return []

    contracts = []
    for m in data.get("markets", []):
        strike_type = m.get("strike_type")
        if strike_type not in ("greater", "less"):
            continue
        contracts.append({
            "ticker":       m["ticker"],
            "series":       series_ticker,
            "city":         city,
            "strike_type":  strike_type,
            "floor_strike": m.get("floor_strike"),
            "cap_strike":   m.get("cap_strike"),
            "close_time":   m.get("close_time"),
            "yes_bid":      m.get("yes_bid"),
            "yes_ask":      m.get("yes_ask"),
        })
    return contracts


def load_all_active_contracts(stations_cfg: dict) -> dict[str, list[dict]]:
    """
    Returns {metar_station: [contract, ...]} for every configured city.
    """
    by_station: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for key, cfg in stations_cfg.items():
        series  = cfg.get("kalshi_series", "")
        metar   = cfg["metar_station"]
        city    = cfg["city"]
        if not series:
            continue
        contracts = fetch_active_contracts(series, city)
        by_station[metar].extend(contracts)
        total += len(contracts)
        time.sleep(0.15)
    print(f"  Active contracts loaded: {total} across {len(by_station)} stations")
    return dict(by_station)


# ── Alert ─────────────────────────────────────────────────────────────────────

def _chime() -> None:
    if SOUND_ENABLED and sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &")


def emit_alert(event: LockEvent, metar_station: str) -> None:
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    bar = "=" * 60

    if event.lock_type == "LOW_confirmed":
        print(f"\n{bar}")
        print(f"  🌡  CONFIRMED LOW — {event.city}  [{now_str}]")
        print(f"  Daily low locked at {event.running_low_f:.1f}°F")
        print(f"  {event.condition}")
        print(bar)
    else:
        side   = "YES" if event.lock_type == "YES_locked" else "NO"
        action = "BUY YES" if event.lock_type == "YES_locked" else "BUY NO"
        print(f"\n{bar}")
        print(f"  *** LOCK {side} — {event.city}  [{now_str}]")
        print(f"  Ticker:    {event.ticker}")
        print(f"  Condition: {event.condition}")
        print(f"  Running high: {event.running_high_f:.1f}°F  |  Running low: {event.running_low_f:.1f}°F")
        print(f"  >> Consider: {action} on Kalshi")
        print(bar)

    _chime()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    stations_cfg = load_stations()

    # Filter cities if requested
    if CITY_FILTER:
        stations_cfg = {k: v for k, v in stations_cfg.items() if k in CITY_FILTER}

    print("=" * 60)
    print("  Kalshi Live METAR Lock Monitor")
    print(f"  Cities: {', '.join(stations_cfg.keys())}")
    print(f"  Poll interval: {POLL_INTERVAL}s  |  Market refresh: {REFRESH_INTERVAL}s")
    print(f"  Sound alerts: {'on' if SOUND_ENABLED else 'off'}")
    print("=" * 60)

    # Build DailyTrackers — one per station
    trackers: dict[str, DailyTracker] = {
        cfg["metar_station"]: DailyTracker(
            station=cfg["metar_station"],
            lst_offset=cfg["lst_offset_hours"],
        )
        for cfg in stations_cfg.values()
    }

    all_stations  = [cfg["metar_station"] for cfg in stations_cfg.values()]
    track_dates:  dict[str, object] = {}   # metar_station → current settlement date

    # Initial market load
    print("\nLoading active contracts from Kalshi…")
    contracts_by_station = load_all_active_contracts(stations_cfg)
    last_refresh = time.monotonic()

    print("\nStarting poll loop. Press Ctrl+C to stop.\n")

    while True:
        utc_now = datetime.now(timezone.utc)

        # ── Reset trackers at midnight LST ───────────────────────────────────
        for station, tracker in trackers.items():
            new_date = tracker.settlement_date(utc_now)
            old_date = track_dates.get(station)
            if old_date is not None and new_date != old_date:
                print(f"[{utc_now:%H:%M UTC}] Day rollover for {station} — resetting tracker")
                tracker.reset()
            track_dates[station] = new_date

        # ── Refresh active markets every REFRESH_INTERVAL ───────────────────
        elapsed = time.monotonic() - last_refresh
        if elapsed >= REFRESH_INTERVAL:
            print(f"\n[{utc_now:%H:%M UTC}] Refreshing active contracts…")
            contracts_by_station = load_all_active_contracts(stations_cfg)
            last_refresh = time.monotonic()

        # ── Fetch METAR for all stations in one call ─────────────────────────
        try:
            obs = fetch_metar(all_stations)
        except Exception as e:
            print(f"[{utc_now:%H:%M UTC}] METAR fetch error: {e}")
            time.sleep(60)
            continue

        # ── Update trackers and check locks ──────────────────────────────────
        status_parts = []
        for station, tracker in trackers.items():
            reading = obs.get(station)
            if reading is None:
                status_parts.append(f"{station}:—")
                continue

            temp_f = reading["temp_f"]
            tracker.update(temp_f)
            status_parts.append(f"{station}:{temp_f:.0f}°F")

            contracts = contracts_by_station.get(station, [])
            for event in tracker.check_locks(contracts, utc_now):
                emit_alert(event, station)

        # ── Status line ──────────────────────────────────────────────────────
        print(f"[{utc_now:%H:%M UTC}]  {' | '.join(status_parts)}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
