"""
Fetch current METAR observations from the NOAA Aviation Weather Center API.

Endpoint: https://aviationweather.gov/api/data/metar
  - Free, no auth required
  - Returns the most recent METAR for each requested station
  - Supports comma-separated station IDs for bulk fetches
  - Temperature reported in Celsius; we convert to Fahrenheit

Typical METAR update frequency: every 20-60 min (ASOS stations hourly,
some report special observations on significant changes).
"""
from __future__ import annotations

import re as _re
from datetime import date, datetime, timedelta

from kalshi_weather.tz import UTC

import time

import requests
import urllib3

_BASE = "https://aviationweather.gov/api/data/metar"
# IEM ASOS archive — returns the full intraday obs history for a station-day,
# unlike the AWC endpoint (latest ob only). Used for one-shot running-high lookups.
_IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_TIMEOUT = 15
_RETRIES = 3


def fetch_running_high_f(
    station: str,
    lst_offset: int,
    on_date: date | None = None,
) -> float | None:
    """
    Running daily HIGH (°F) observed so far for `station` over its Local-Standard
    day, from IEM ASOS observations.

    The live dashboard accumulates the running high by polling METAR continuously;
    a one-shot cron run has no such history, so it queries the full LST-day record
    here instead. Day boundary is LST (Kalshi settles on LST, not the DST clock).

    station    : ICAO id (e.g. "KHOU"); a leading "K" is stripped for the US ASOS net.
    lst_offset : hours from UTC for Local Standard Time (e.g. -6 for CST).
    on_date    : LST calendar date to query; defaults to "today" in LST.

    Returns the running high °F, or None on fetch failure / no observations (the
    caller treats None as "no lock data available", never as a lock).
    """
    iem_id = station[1:] if (len(station) == 4 and station.upper().startswith("K")) else station
    if lst_offset < 0:
        tz = f"Etc/GMT+{abs(lst_offset)}"
    elif lst_offset > 0:
        tz = f"Etc/GMT-{lst_offset}"
    else:
        tz = "UTC"
    if on_date is None:
        on_date = (datetime.now(UTC) + timedelta(hours=lst_offset)).date()

    params = {
        "station": iem_id, "data": "tmpf", "tz": tz, "format": "onlycomma",
        "missing": "empty", "latlon": "no",
        "year1": on_date.year, "month1": on_date.month, "day1": on_date.day,
        "year2": on_date.year, "month2": on_date.month, "day2": on_date.day,
    }
    # IEM rate-limits rapid requests (HTTP 429); a one-shot cron fetching ~20
    # stations back-to-back trips it. Retry with backoff so a throttle doesn't
    # masquerade as "no obs" (which would silently skip lock detection).
    resp = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(_IEM_ASOS_URL, params=params, timeout=_TIMEOUT)
            if resp.status_code == 429:
                raise requests.exceptions.HTTPError("429 rate-limited")
            resp.raise_for_status()
            break
        except Exception:
            resp = None
            if attempt < _RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    if resp is None:
        return None

    highs: list[float] = []
    for line in resp.text.splitlines()[1:]:          # skip header
        parts = line.split(",")
        if len(parts) >= 3 and parts[2].strip():
            try:
                highs.append(float(parts[2]))
            except ValueError:
                pass
    return max(highs) if highs else None

# ---------------------------------------------------------------------------
# Weather risk parsing
# ---------------------------------------------------------------------------

_WIND_RE = _re.compile(r'^(?:VRB|\d{3})(\d{2,3})(?:G(\d{2,3}))?KT$')
_SKY_RE  = _re.compile(r'^(CLR|SKC|CAVOK|FEW|SCT|BKN|OVC|VV)(\d{3})?')
_SKY_RANK = {'CLR': 0, 'SKC': 0, 'CAVOK': 0, 'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4, 'VV': 4}

# Base weather-phenomenon codes (without intensity prefix)
_WX_ALL = {
    'TS', 'TSRA', 'TSGR', 'TSSN', 'TSGS', 'TSPL',
    'RA', 'SN', 'DZ', 'GR', 'GS', 'SG', 'PL', 'IC',
    'SHRA', 'SHSN', 'SHGR', 'SHGS', 'SHDZ',
    'FZRA', 'FZDZ', 'FZFG',
    'FG', 'BCFG', 'MIFG', 'PRFG',
    'BR', 'HZ', 'FU', 'DU', 'SA', 'PO', 'SQ', 'FC', 'SS', 'DS', 'VA',
    'BLSN', 'BLDU', 'BLSA',
}
_TS_CODES  = {'TS', 'TSRA', 'TSGR', 'TSSN', 'TSGS', 'TSPL'}
_FZ_CODES  = {'FZRA', 'FZDZ', 'FZFG'}
_FOG_CODES = {'FG', 'BCFG', 'MIFG', 'PRFG', 'FZFG'}
_RAIN_CODES = {'RA', 'SHRA', 'TSRA', 'FZRA', 'DZ', 'SHDZ', 'FZDZ'}
_SNOW_CODES = {'SN', 'SHSN', 'BLSN', 'TSSN', 'GS', 'GR', 'PL'}


def parse_metar_risk(raw: str) -> dict:
    """
    Parse a raw METAR string and return a weather risk assessment.

    Returns a dict with keys:
      level      : "safe" | "watch" | "danger"
      tags       : list[str]  — human-readable condition labels
      wind_kt    : int | None
      phenomena  : list[str]  — raw wx tokens found
      ceiling_ft : int | None
    """
    tokens = raw.upper().split()

    wind_kt     = None
    phenomena   = []
    worst_cover = 0
    ceiling_ft  = None

    for tok in tokens:
        m = _WIND_RE.match(tok)
        if m:
            speed = int(m.group(1))
            gust  = int(m.group(2)) if m.group(2) else speed
            wind_kt = max(speed, gust)
            continue

        m = _SKY_RE.match(tok)
        if m:
            cover   = m.group(1)
            h_hund  = m.group(2)
            h_ft    = int(h_hund) * 100 if h_hund else None
            rank    = _SKY_RANK.get(cover, 0)
            if rank > worst_cover:
                worst_cover = rank
            if cover in ('BKN', 'OVC', 'VV') and h_ft is not None:
                if ceiling_ft is None or h_ft < ceiling_ft:
                    ceiling_ft = h_ft
            continue

        # Weather phenomena: strip intensity prefix (+/-) and VC for lookup
        bare = tok.lstrip('+-')
        if bare.startswith('VC'):
            bare = bare[2:]
        if tok in _WX_ALL or bare in _WX_ALL:
            phenomena.append(tok)

    # Classify phenomena
    bare_set   = {p.lstrip('+-') for p in phenomena}
    bare_set   = {p[2:] if p.startswith('VC') else p for p in bare_set}
    has_heavy  = any(p.startswith('+') for p in phenomena)
    has_ts     = bool(bare_set & _TS_CODES)
    has_fz     = bool(bare_set & _FZ_CODES)
    has_fog    = bool(bare_set & _FOG_CODES)
    has_rain   = bool(bare_set & _RAIN_CODES)
    has_snow   = bool(bare_set & _SNOW_CODES)

    # Human-readable tags (shown on dashboard)
    tags: list[str] = []
    if has_ts:    tags.append('TS')
    if has_fz:    tags.append('FZPRECIP')
    if has_heavy: tags.append('+PRECIP')
    if has_fog:   tags.append('FG')
    if has_rain and not has_ts:  tags.append('RAIN')
    if has_snow and not has_ts:  tags.append('SNOW')
    if wind_kt and wind_kt >= 30:
        tags.append(f'WIND {wind_kt}kt')
    elif wind_kt and wind_kt >= 20:
        tags.append(f'WIND {wind_kt}kt')
    if ceiling_ft is not None and ceiling_ft < 1500:
        tags.append(f'CEIL {ceiling_ft}ft')
    elif worst_cover >= 3 and not tags:
        tags.append('OVC')

    # Risk level
    if has_ts or has_fz or has_heavy or (wind_kt is not None and wind_kt >= 35):
        level = 'danger'
    elif (has_rain or has_snow or has_fog
          or (wind_kt is not None and wind_kt >= 20)
          or worst_cover >= 3
          or (ceiling_ft is not None and ceiling_ft < 3000)):
        level = 'watch'
    else:
        level = 'safe'

    return {
        'level':      level,
        'tags':       tags,
        'wind_kt':    wind_kt,
        'phenomena':  phenomena,
        'ceiling_ft': ceiling_ft,
    }


def fetch_metar(stations: list[str]) -> dict[str, dict]:
    """
    Fetch the most recent METAR observation for each station in the list.

    Parameters
    ----------
    stations : list of ICAO station IDs (e.g. ["KNYC", "KORD", "KMIA"])

    Returns
    -------
    dict mapping station_id → observation dict with keys:
        station, temp_f, obs_time (UTC datetime), raw, weather_risk
    Stations with no data are omitted from the result.
    """
    ids = ",".join(s.upper() for s in stations)
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(
                _BASE,
                params={"ids": ids, "format": "geojson", "taf": "false"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            break
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            urllib3.exceptions.ProtocolError,
        ) as exc:
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(2 ** attempt)
    else:
        raise last_exc  # type: ignore[misc]

    result: dict[str, dict] = {}
    for feature in resp.json().get("features", []):
        props = feature.get("properties", {})
        raw   = props.get("rawOb", "")

        # rawOb format: "METAR KNYC ..." or "KNYC ..." — skip type prefix
        tokens = raw.split()
        if tokens and tokens[0].upper() in ("METAR", "SPECI"):
            station = tokens[1] if len(tokens) > 1 else ""
        elif tokens:
            station = tokens[0]
        else:
            station = props.get("site", "")

        if not station:
            continue

        temp_c = props.get("temp")
        if temp_c is None:
            continue

        obs_str = props.get("obsTime", "")
        try:
            obs_time = datetime.fromisoformat(obs_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            obs_time = datetime.now(UTC)

        result[station] = {
            "station":      station,
            "temp_f":       round(float(temp_c) * 9 / 5 + 32, 1),
            "temp_c":       float(temp_c),
            "obs_time":     obs_time,
            "raw":          raw,
            "weather_risk": parse_metar_risk(raw),
        }

    return result
