"""
Multi-model forecast archive — point-in-time snapshots of every model per city-day.

Right now each model's forecast is fetched, used, and thrown away — so we can never learn
which model is actually best per city (Miami: GEFS says 88.8°, ICON says 93.3° — which is
right? unknown). This archives GFS / ECMWF / NBM / HRRR / NWS for all 20 cities, for the next
3 settlement days, every run — so each settlement date is captured at multiple lead times.
Joined to the actual CLI high later (scripts/model_skill.py) it becomes a which-model-wins
dataset and a basis for a data-driven blend, and it multiplies learning per day (20 cities ×
several models × 3 leads per run vs the handful we trade).

Run a few times a day via cron. Appends to data/logger/forecasts/forecasts.jsonl.
Cheap: 20 cities × 4 models ≈ 80 calls/run, far under the 10k/day free tier.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalshi_weather.config import load_stations           # noqa: E402
from kalshi_weather.calibration.ensemble_dist import fetch_nws_high  # noqa: E402

OUT = ROOT / "data" / "logger" / "forecasts" / "forecasts.jsonl"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
# "gfs" must be gfs_global (pure GFS): gfs_seamless IS HRRR for the first ~2 days,
# which made the gfs and hrrr columns identical at short leads. gfs_model tags rows
# so model_skill.py can exclude the contaminated legacy column.
DET_MODELS = {"gfs": "gfs_global", "ecmwf": "ecmwf_ifs025",
              "nbm": "ncep_nbm_conus", "hrrr": "gfs_hrrr"}
FORECAST_DAYS = 3


def _fetch_model(lat, lon, tz, model_id) -> dict[str, float]:
    """{settlement_date -> temperature_2m_max} for one model, next FORECAST_DAYS days."""
    params = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
              "temperature_unit": "fahrenheit", "timezone": tz,
              "forecast_days": FORECAST_DAYS, "models": model_id}
    for attempt in range(3):
        try:
            r = requests.get(FORECAST_API, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2); continue
            r.raise_for_status()
            daily = r.json().get("daily", {})
            out = {}
            for d, t in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
                if t is not None:
                    out[d] = round(float(t), 1)
            return out
        except Exception:
            time.sleep(1)
    return {}


def main() -> None:
    stations = load_stations()
    ts = datetime.now(timezone.utc).isoformat()
    today = date.today()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "a") as fh:
        for cfg in stations.values():
            lat, lon, tz = cfg["lat"], cfg["lon"], cfg["timezone"]
            station = cfg["metar_station"]
            per_model = {name: _fetch_model(lat, lon, tz, mid) for name, mid in DET_MODELS.items()}
            time.sleep(0.3)  # stay polite to the API
            # union of dates any model returned
            dates = sorted({d for m in per_model.values() for d in m})
            for d in dates:
                rec = {
                    "snapshot_ts": ts, "station": station, "city": cfg.get("city"),
                    "settlement_date": d,
                    "lead_days": (date.fromisoformat(d) - today).days,
                    "gfs": per_model["gfs"].get(d), "gfs_model": DET_MODELS["gfs"],
                    "ecmwf": per_model["ecmwf"].get(d),
                    "nbm": per_model["nbm"].get(d), "hrrr": per_model["hrrr"].get(d),
                    "nws": fetch_nws_high(lat, lon, d),
                }
                fh.write(json.dumps(rec) + "\n")
                n += 1
    print(f"archived {n} city-date forecast rows → {OUT}")


if __name__ == "__main__":
    # Once-per-window guard so the cron entry and the launchd agent (catch-up on wake)
    # can't double-fire. Scheduled ~6h apart; the 30-min gap only trips a duplicate.
    from kalshi_weather.scheduling import acquire_slot, mark_and_release
    _state = ROOT / "logs" / "scheduler"
    _slot = acquire_slot(_state, "archive_forecasts", min_gap_s=1800, force="--force" in sys.argv)
    if _slot is None:
        print("archive_forecasts: ran recently or another instance is active — skipping.")
        sys.exit(0)
    try:
        main()
    finally:
        mark_and_release(_state, "archive_forecasts", _slot)
