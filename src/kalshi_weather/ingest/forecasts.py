"""
Fetch and cache weather model forecast data from Open-Meteo Historical Forecast API.

Returns the reconstructed "what was forecast for this date" tmax, which is the
closest proxy available without individual ensemble member archives pre-2025.

In Milestone 2, this will be replaced by run-specific forecasts with uncertainty.
For Milestone 1, we use the forecast tmax as input to the calibration error model.

Cache layout: data/raw/forecasts/{model}/{year}.parquet

Rate limits (free tier): 10,000 calls/day; requests >10 variables or >2 weeks
count as multiple calls. We fetch one variable at a time per year chunk.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from kalshi_weather.tz import utc_now

_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_CACHE_ROOT = Path(__file__).parents[3] / "data" / "raw" / "forecasts"

# Models available in the historical forecast API with their approximate start dates
MODELS = {
    "gfs_seamless":  "2021-01-01",   # GFS from 2021 — covers full Kalshi history
    "ecmwf_ifs025":  "2024-02-15",   # ECMWF IFS025 from ~Feb 2024 (verified empirically)
}

_CHUNK_DAYS = 90  # stay well under the 2-week multiple-call threshold


def _cache_path(model: str, year: int, station: str = "KNYC") -> Path:
    # Legacy path (no station prefix) is accepted for KNYC to preserve existing cache.
    # All new files use station-prefixed names.
    new_path = _CACHE_ROOT / model / f"{station}_{year}.parquet"
    if new_path.exists():
        return new_path
    legacy = _CACHE_ROOT / model / f"{year}.parquet"
    if legacy.exists() and station == "KNYC":
        return legacy
    return new_path


def _fetch_chunk(
    lat: float,
    lon: float,
    model: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start,
        "end_date":        end,
        "daily":           "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone":        "UTC",
        "models":          model,
    }
    resp = requests.get(_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    df = pd.DataFrame({
        "date":        [d.date() for d in pd.to_datetime(daily["time"])],
        "tmax_f_fcst": pd.to_numeric(daily.get("temperature_2m_max"), errors="coerce"),
        "tmin_f_fcst": pd.to_numeric(daily.get("temperature_2m_min"), errors="coerce"),
    })
    return df


def fetch_forecasts(
    station: str,
    lat: float,
    lon: float,
    years: list[int],
    models: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch and cache daily tmax/tmin forecasts for the given coordinates and years.
    Returns a DataFrame with columns: station, model, date, tmax_f_fcst, tmin_f_fcst.
    """
    if models is None:
        models = list(MODELS.keys())

    frames: list[pd.DataFrame] = []

    for model in models:
        (_CACHE_ROOT / model).mkdir(parents=True, exist_ok=True)

        model_start = pd.Timestamp(MODELS[model])

        for year in sorted(years):
            path = _cache_path(model, year, station)

            if path.exists() and not force_refresh:
                frames.append(pd.read_parquet(path))
                continue

            year_start = max(pd.Timestamp(f"{year}-01-01"), model_start)
            year_end   = pd.Timestamp(f"{year}-12-31")
            chunks: list[pd.DataFrame] = []

            cursor = year_start
            while cursor <= year_end:
                chunk_end = min(cursor + pd.Timedelta(days=_CHUNK_DAYS - 1), year_end)
                try:
                    chunk_df = _fetch_chunk(
                        lat, lon, model,
                        cursor.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d"),
                    )
                    chunks.append(chunk_df)
                except requests.HTTPError as e:
                    print(f"  Warning: {model} {year} chunk {cursor.date()} failed: {e}")
                cursor = chunk_end + pd.Timedelta(days=1)
                time.sleep(0.15)

            if not chunks:
                continue

            df = pd.concat(chunks, ignore_index=True)
            df["station"] = station
            df["model"] = model
            df["retrieved_utc"] = utc_now().isoformat()
            save_path = _CACHE_ROOT / model / f"{station}_{year}.parquet"
            df.to_parquet(save_path, index=False)
            frames.append(df)
            print(f"  Cached {model} {year}: {len(df)} days")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["model", "date"]).reset_index(drop=True)
