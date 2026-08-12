"""
Logs Open-Meteo GEFS and ECMWF ensemble members to parquet.

Why this matters: the public Open-Meteo API only retains individual ensemble
members for ~3 days. After that, only the ensemble mean is accessible. This
logger captures every member of every model run from today forward, building
a proprietary dataset that cannot be reconstructed from any public API later.

Storage layout:
  data/logger/ensemble/{model}/{YYYYMMDD_HH}Z/{station}.parquet

Schema per file:
  init_time   datetime64[ns, UTC]  — when this model run was initialized
  valid_time  datetime64[ns, UTC]  — the hour this row forecasts
  member      int16                — member index (0 = control run)
  temp_f      float32              — 2m temperature in °F for that hour

Model run cadences:
  GEFS (gfs025):      00Z / 06Z / 12Z / 18Z  (31 members: control + 30 perturbed)
  ECMWF (ecmwf_ifs04): 00Z / 12Z             (51 members: control + 50 perturbed)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from kalshi_weather.tz import UTC
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

# Ensemble models to log. Keys are Open-Meteo model identifiers.
MODELS: dict[str, dict] = {
    "gfs025": {
        "label":      "GEFS",
        "interval_h": 6,    # new run every 6 hours (00/06/12/18Z)
        "n_members":  30,   # 30 perturbed members (no control on free tier)
        "lag_h":      5,    # typically available ~5h after init time
    },
    "ecmwf_ifs025": {
        "label":      "ECMWF ENS",
        "interval_h": 12,   # new run every 12 hours (00/12Z)
        "n_members":  50,   # 50 perturbed members on free tier
        "lag_h":      8,    # typically available ~8h after init time
    },
    "icon_seamless": {
        "label":      "ICON ENS",
        "interval_h": 6,    # new run every 6 hours
        "n_members":  40,   # ~39-40 members
        "lag_h":      4,
    },
}

_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_FORECAST_DAYS = 3   # today + 2 more days; enough for daily high contract coverage


def _init_times_to_fetch(model_key: str, utc_now: datetime) -> list[datetime]:
    """
    Return init_times that should be available by now and haven't been stored yet.
    We look back 2 complete intervals to catch any run we might have missed.
    """
    cfg = MODELS[model_key]
    interval = cfg["interval_h"]
    lag = cfg["lag_h"]

    available_by = utc_now - timedelta(hours=lag)
    h = (available_by.hour // interval) * interval
    latest = available_by.replace(hour=h, minute=0, second=0, microsecond=0)

    return [latest - timedelta(hours=interval * i) for i in range(2)]


def _cache_path(log_dir: Path, model_key: str, init_time: datetime, station: str) -> Path:
    run_label = init_time.strftime("%Y%m%d_%H") + "Z"
    return log_dir / model_key / run_label / f"{station}.parquet"


def fetch_missing(stations_cfg: dict, log_dir: Path) -> int:
    """
    Check for ensemble runs not yet stored and fetch them.
    Returns count of new run/station combinations fetched.
    """
    fetched = 0
    utc_now = datetime.now(UTC)

    for model_key, model_cfg in MODELS.items():
        for init_time in _init_times_to_fetch(model_key, utc_now):
            for cfg in stations_cfg.values():
                station = cfg["metar_station"]
                path = _cache_path(log_dir, model_key, init_time, station)
                if path.exists():
                    continue

                try:
                    df = _fetch_one(model_key, init_time, cfg)
                    if df is not None and not df.empty:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(path, index=False)
                        fetched += 1
                        log.debug("stored %s %s %s (%d rows)",
                                  model_cfg["label"], init_time.strftime("%Y-%m-%d %HZ"),
                                  station, len(df))
                except Exception as exc:
                    log.warning("ensemble fetch failed %s %s %s: %s",
                                model_key, init_time.strftime("%HZ"), station, exc)

                time.sleep(0.5)   # stay within Open-Meteo free-tier rate limit

    return fetched


def _fetch_one(model_key: str, init_time: datetime, station_cfg: dict) -> pd.DataFrame | None:
    """
    Fetch one model run for one station from Open-Meteo ensemble API.
    Returns a tidy DataFrame with columns: init_time, valid_time, member, temp_f.
    """
    params = {
        "latitude":         station_cfg["lat"],
        "longitude":        station_cfg["lon"],
        "hourly":           "temperature_2m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "timezone":         "UTC",
        "models":           model_key,
        "forecast_days":    _FORECAST_DAYS,
    }

    resp = requests.get(_ENSEMBLE_URL, params=params, timeout=30)
    if resp.status_code == 429:
        log.warning("rate limited by Open-Meteo; backing off 30s")
        time.sleep(30)
        resp = requests.get(_ENSEMBLE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return None

    valid_times = pd.to_datetime(times, utc=True)

    # Collect all member columns: temperature_2m_member0, _member01, _member02, ...
    member_cols = [k for k in hourly if k.startswith("temperature_2m_member")]
    if not member_cols:
        log.warning("no member columns found for %s %s", model_key, station_cfg["metar_station"])
        return None

    rows: list[dict] = []
    for col in member_cols:
        # Parse member index: "temperature_2m_member0" → 0, "temperature_2m_member01" → 1
        suffix = col.replace("temperature_2m_member", "")
        try:
            member_id = int(suffix)
        except ValueError:
            continue

        values = hourly[col]
        for valid_time, temp_f in zip(valid_times, values):
            if temp_f is None:
                continue
            rows.append({
                "init_time":  init_time,
                "valid_time": valid_time,
                "member":     member_id,
                "temp_f":     float(temp_f),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["init_time"]  = pd.to_datetime(df["init_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["member"]     = df["member"].astype("int16")
    df["temp_f"]     = df["temp_f"].astype("float32")

    return df


def daily_high_distribution(
    log_dir: Path,
    station: str,
    settlement_date: "date",
    lst_offset_h: int,
    model_key: str | None = None,
    max_age_h: int = 12,
) -> pd.DataFrame | None:
    """
    For a given settlement date, load all stored ensemble members and compute
    the per-member daily max temperature over the settlement window.

    Args:
        log_dir:         root logger directory
        station:         METAR station ID (e.g. 'KNYC')
        settlement_date: the date whose tmax we're forecasting
        lst_offset_h:    LST offset (e.g. -5 for NYC); defines the day boundary
        model_key:       filter to one model (None = all available)
        max_age_h:       ignore model runs older than this many hours

    Returns a DataFrame with columns: model, init_time, member, daily_high_f
    or None if no data found.
    """
    from datetime import date

    # Settlement window in UTC: midnight LST → midnight LST next day
    day_start_utc = datetime(
        settlement_date.year, settlement_date.month, settlement_date.day,
        tzinfo=UTC,
    ) + timedelta(hours=-lst_offset_h)
    day_end_utc = day_start_utc + timedelta(hours=24)

    cutoff_utc = datetime.now(UTC) - timedelta(hours=max_age_h)
    models_to_check = [model_key] if model_key else list(MODELS)

    frames: list[pd.DataFrame] = []
    for mk in models_to_check:
        model_dir = log_dir / mk
        if not model_dir.exists():
            continue
        for run_dir in sorted(model_dir.iterdir()):
            # run_dir name: YYYYMMDD_HHZ
            try:
                init_dt = datetime.strptime(run_dir.name, "%Y%m%d_%HZ").replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue
            if init_dt < cutoff_utc:
                continue

            path = run_dir / f"{station}.parquet"
            if not path.exists():
                continue

            df = pd.read_parquet(path)
            window = df[
                (df["valid_time"] >= day_start_utc) &
                (df["valid_time"] <  day_end_utc)
            ]
            if window.empty:
                continue

            daily_max = (
                window.groupby("member")["temp_f"]
                .max()
                .reset_index()
                .rename(columns={"temp_f": "daily_high_f"})
            )
            daily_max["model"]     = mk
            daily_max["init_time"] = init_dt
            frames.append(daily_max)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)
