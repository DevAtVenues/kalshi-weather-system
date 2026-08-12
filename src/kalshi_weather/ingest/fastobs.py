"""
Fast-cadence station observations for boundary watching.

Two free feeds, merged:
  1. api.weather.gov /stations/{id}/observations — 5-MINUTE ASOS obs, but whole
     degrees CELSIUS ("37 C" means true temp anywhere in 97.7-99.5 F), and
     ~10-25 min publication latency. 12x the cadence of METARs.
  2. AWC METARs (aviationweather.gov) — hourly, but the decoded T-group carries
     TENTHS (36.7 C = 98.1 F). Precision truth, slow cadence.

Learned live 2026-07-21 (SAT B98.5): the 5-min integer-C feed printed "98.6 F"
for an hour while the hourly tenths said 98.1 flat — integer-C rounding near a
bracket boundary is a full degree of illusion, and the market's bots read the
tenths. Every consumer must treat an integer-C row as a BAND
[(c-0.5), (c+0.5)] C, never as a point.

Merged result per station-LST-day (°F):
  high_min_f : floor on the true max — tenths rows count exactly, integer-C
               rows contribute their band LOWER edge
  high_max_f : how hot it might already have been — integer-C band UPPER edge
The CLI settlement high can still exceed high_min_f: it is the max of the
CONTINUOUS one-minute record, and both of these feeds are snapshots.

ADVISORY consumer only. Any failure returns None/[] — never raises into a run.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import requests

from kalshi_weather.tz import utc_now       # single tz seam (leakage-tested)

_NWS_OBS_URL = "https://api.weather.gov/stations/{station}/observations"
_AWC_URL = "https://aviationweather.gov/api/data/metar"
_UA = {"User-Agent": "kalshi-weather-research (fastobs)"}
_TIMEOUT = (10, 30)

# A reading whose Celsius value has a non-zero tenths digit came from a decoded
# T-group (precise). Integer values are band-rounded. (A true x.0 tenths reading
# is misclassified as a band — conservative, never optimistic.)
_PRECISE_EPS = 1e-6


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def is_precise_c(temp_c: float) -> bool:
    return abs(temp_c * 10.0 - round(temp_c * 10.0)) < _PRECISE_EPS and \
        abs(temp_c - round(temp_c)) > _PRECISE_EPS


def fetch_nws_obs(station: str, limit: int = 72) -> list[dict]:
    """5-min feed rows: [{ts, temp_c, raw}] newest first. [] on any failure."""
    try:
        resp = requests.get(_NWS_OBS_URL.format(station=station),
                            params={"limit": limit}, headers=_UA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return []
        out = []
        for f in resp.json().get("features", []):
            p = f.get("properties", {})
            t = (p.get("temperature") or {}).get("value")
            ts = p.get("timestamp")
            if ts is None:
                continue
            out.append({"ts": pd.Timestamp(ts), "temp_c": t,
                        "raw": p.get("rawMessage") or ""})
        return out
    except Exception:
        return []


def fetch_awc_metars(station: str, hours: int = 10) -> list[dict]:
    """AWC decoded METARs (temp in tenths C when T-group present). [] on failure."""
    try:
        resp = requests.get(_AWC_URL, params={"ids": station, "format": "json",
                                              "hours": hours},
                            headers=_UA, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return []
        out = []
        for o in resp.json():
            t = o.get("temp")
            ts = o.get("reportTime") or o.get("obsTime")
            if ts is None or t is None:
                continue
            if isinstance(ts, (int, float)):        # obsTime is epoch seconds
                ts = pd.Timestamp(float(ts), unit="s", tz="UTC")
            else:
                ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")          # AWC times are UTC by spec
            out.append({"ts": ts, "temp_c": float(t), "raw": o.get("rawOb") or ""})
        return out
    except Exception:
        return []


def band_from_rows(rows: list[dict], lst_offset_h: int,
                   on_date: date, ) -> dict | None:
    """Merge observation rows into the day's running band. Pure — no network.

    rows: [{ts: tz-aware Timestamp, temp_c: float|None, raw: str}]
    Rows outside the station's LST day `on_date` are ignored.
    """
    offset = timedelta(hours=lst_offset_h)
    precise: list[tuple[pd.Timestamp, float]] = []   # (ts, temp_f)
    high_min = None
    high_max = None
    n = 0
    for r in rows:
        t = r.get("temp_c")
        ts = r.get("ts")
        if t is None or ts is None:
            continue
        if (ts + offset).date() != on_date:
            continue
        n += 1
        f = c_to_f(float(t))
        if is_precise_c(float(t)):
            precise.append((ts, f))
            lo = hi = f
        else:                                        # integer-C band
            lo, hi = c_to_f(float(t) - 0.5), c_to_f(float(t) + 0.5)
        high_min = lo if high_min is None else max(high_min, lo)
        high_max = hi if high_max is None else max(high_max, hi)
    if n == 0:
        return None
    precise.sort(key=lambda x: x[0], reverse=True)
    latest_precise = precise[0] if precise else None
    flat_minutes = None
    if latest_precise:
        flat_minutes = 0.0
        for ts, f in precise[1:]:
            if abs(f - latest_precise[1]) > 0.25:
                break
            flat_minutes = (latest_precise[0] - ts).total_seconds() / 60.0
    return {
        "high_min_f": round(high_min, 2),
        "high_max_f": round(high_max, 2),
        "latest_ts": max(r["ts"] for r in rows if r.get("temp_c") is not None
                         and (r["ts"] + offset).date() == on_date),
        "latest_precise_f": round(latest_precise[1], 2) if latest_precise else None,
        "flat_minutes": None if flat_minutes is None else round(flat_minutes),
        "n_obs": n,
    }


def day_running_band(station: str, lst_offset_h: int,
                     on_date: date | None = None,
                     now: datetime | None = None) -> dict | None:
    """Merged fast-obs running band for (station, LST day). None on no data."""
    now = now or utc_now()
    if on_date is None:
        on_date = (pd.Timestamp(now) + timedelta(hours=lst_offset_h)).date()
    rows = fetch_nws_obs(station) + fetch_awc_metars(station)
    return band_from_rows(rows, lst_offset_h, on_date)
