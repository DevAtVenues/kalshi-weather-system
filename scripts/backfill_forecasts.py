"""
Backfill the multi-model archive from Open-Meteo's Historical FORECAST API.

Gives a which-model-wins answer NOW from months of history, instead of waiting for the
forward archive to fill. Pulls the archived GFS/ECMWF/NBM/HRRR forecast (as issued, ~1-day
lead — this is the Historical *Forecast* API, NOT reanalysis, so it's a fair skill measure,
not look-ahead) for every city over a date range, and writes it where model_skill.py reads.

Caveats (stated, not hidden): the archive's exact lead time isn't guaranteed to be exactly 1
day, so absolute MAE may be a touch optimistic — but all models are pulled identically, so
the RELATIVE ranking (which we use for blend weights) is sound. Writes to a SEPARATE file
(overwrite) so re-runs don't bloat the forward archive; forward snapshots win on overlap.

Usage: .venv/bin/python scripts/backfill_forecasts.py [--days 120]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kalshi_weather.config import load_stations  # noqa: E402

OUT = ROOT / "data" / "logger" / "forecasts" / "forecasts_backfill.jsonl"
HIST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# "gfs" must be gfs_global (pure GFS): gfs_seamless IS HRRR for the first ~2 days,
# which made the gfs and hrrr columns identical at short leads. gfs_model tags rows
# so model_skill.py can exclude the contaminated legacy column.
DET_MODELS = {"gfs": "gfs_global", "ecmwf": "ecmwf_ifs025",
              "nbm": "ncep_nbm_conus", "hrrr": "gfs_hrrr"}
# sorts BEFORE any real ISO snapshot_ts, so a forward archive row always wins the dedup
BACKFILL_TS = "0000-01-01T00:00:00+00:00"


def _fetch(lat, lon, tz, model_id, start, end) -> dict[str, float]:
    p = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
         "daily": "temperature_2m_max", "temperature_unit": "fahrenheit",
         "timezone": tz, "models": model_id}
    for attempt in range(3):
        try:
            r = requests.get(HIST_API, params=p, timeout=45)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2); continue
            r.raise_for_status()
            daily = r.json().get("daily", {})
            return {d: round(float(t), 1) for d, t in
                    zip(daily.get("time", []), daily.get("temperature_2m_max", [])) if t is not None}
        except Exception:
            time.sleep(1)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)
    stations = load_stations()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w") as fh:                      # overwrite — clean re-runs
        for cfg in stations.values():
            lat, lon, tz, stn = cfg["lat"], cfg["lon"], cfg["timezone"], cfg["metar_station"]
            pm = {name: _fetch(lat, lon, tz, mid, start.isoformat(), end.isoformat())
                  for name, mid in DET_MODELS.items()}
            time.sleep(0.4)
            for d in sorted({dd for m in pm.values() for dd in m}):
                fh.write(json.dumps({
                    "snapshot_ts": BACKFILL_TS, "source": "backfill", "lead_days": 1,
                    "station": stn, "city": cfg.get("city"), "settlement_date": d,
                    "gfs": pm["gfs"].get(d), "gfs_model": DET_MODELS["gfs"],
                    "ecmwf": pm["ecmwf"].get(d),
                    "nbm": pm["nbm"].get(d), "hrrr": pm["hrrr"].get(d), "nws": None,
                }) + "\n")
                n += 1
            print(f"  {stn}: {sum(1 for m in pm.values() for _ in m)//max(1,len(pm))} days")
    print(f"backfilled {n} city-date rows ({start}→{end}) → {OUT}")


if __name__ == "__main__":
    main()
