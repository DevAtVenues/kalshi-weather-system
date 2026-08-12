"""
Web dashboard launcher for the Kalshi Weather Monitor.

Opens a browser tab automatically, then serves the dashboard at localhost.
Leave it running in the background; Ctrl+C to stop.

Run:
  .venv/bin/python scripts/run_web_dashboard.py

Options (env vars):
  DASHBOARD_PORT         — port (default: 5555)
  DASHBOARD_POLL_SECONDS — METAR + price refresh interval (default: 120 = 2 min)
  DASHBOARD_CITIES       — comma-separated city keys (default: all)
                           e.g.  DASHBOARD_CITIES=NYC,CHI,MIA
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.config import load_stations
from kalshi_weather.dashboard.app import create_app

PORT         = int(os.getenv("DASHBOARD_PORT",         "5555"))
POLL_INTERVAL = int(os.getenv("DASHBOARD_POLL_SECONDS", "120"))

_CITY_FILTER = os.getenv("DASHBOARD_CITIES", "")
CITY_FILTER  = set(_CITY_FILTER.upper().split(",")) if _CITY_FILTER else set()


def main() -> None:
    stations_cfg = load_stations()
    if CITY_FILTER:
        stations_cfg = {k: v for k, v in stations_cfg.items() if k in CITY_FILTER}

    print("=" * 56)
    print("  Kalshi Weather Dashboard")
    print(f"  Cities : {', '.join(stations_cfg.keys())}")
    print(f"  Poll   : every {POLL_INTERVAL // 60} min")
    print(f"  URL    : http://localhost:{PORT}")
    print("=" * 56)
    print("\nLoading initial data (GFS + Kalshi + METAR)…")

    flask_app = create_app(stations_cfg, POLL_INTERVAL)

    # Open browser after a short delay so Flask is ready first
    def _open():
        time.sleep(2)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=_open, daemon=True).start()

    flask_app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
