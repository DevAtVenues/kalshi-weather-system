"""
NBM (National Blend of Models) station MaxT percentiles from the NBP bulletin.

Why this source: NBM is NOAA's calibrated blend of dozens of models,
statistically corrected TO THE STATION observation record — a free,
professionally-calibrated probability distribution at (approximately) the
settlement sensor, sidestepping the grid→sensor bias problem our ensemble
pipeline corrects by hand. The NBP text product carries QMD MaxT percentiles
(TXNP1/2/5/7/9 = 10/25/50/75/90th) per station.

Source: the operational bulk file on NOMADS (the officially supported path):
  .../blend.{YYYYMMDD}/{CC}/text/blend_nbptx.t{CC}z     (~33 MB, all stations)
Full cycles: 01/07/13/19 UTC, available ~1h after cycle time. We stream the
file ONCE per cycle, extract only our stations, and cache the parsed blocks
to data/cache/nbm/ — the 4x/day board schedule means at most 4 downloads/day.

Format notes (pinned against the live 2026-07-21 01Z file, NBM V5.0):
  KSAT    NBM V5.0 NBP GUIDANCE    7/21/2026  0100 UTC
         WED 22| THU 23| ...
  UTC    00  12| 00  12| ...
  FHR    23  35| 47  59| ...
  TXNP5  97  79| 99  84| ...
Columns are 3-char right-aligned fields; "|" separates day groups. MaxT
values sit in the 00Z columns: the daytime max (7am–7pm LST) is reported at
the ~00Z that ENDS that local day, so a value valid 00Z on UTC day D+1 is
the high for LOCAL day D (verified: KSAT rows alternate 97/79 = max/min).

ADVISORY consumer only (candidate_pipeline vet). Failure of any kind returns
None/{} — never raises into a board run.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from kalshi_weather.tz import utc_now           # single tz seam (leakage-tested)

NBM_URL = ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/"
           "blend.{ymd}/{cyc:02d}/text/blend_nbptx.t{cyc:02d}z")
CACHE_DIR = Path(__file__).parents[3] / "data" / "cache" / "nbm"
FULL_CYCLES = (1, 7, 13, 19)     # UTC cycles with full NBP data
_AVAIL_LAG_H = 1.5               # don't try a cycle younger than this
_LOOKBACK_H = 30.0               # give up beyond this

_PCT_ROWS = {"TXNP1": "p10", "TXNP2": "p25", "TXNP5": "p50",
             "TXNP7": "p75", "TXNP9": "p90", "TXNMN": "mean", "TXNSD": "sd"}

# Matches ANY station header (K/P/T/C ids, buoys…) — the stream splitter must
# close a tracked block at EVERY station boundary, or a following station's
# rows would bleed into ours.
_HDR_RE = re.compile(r"^ ?([A-Z0-9]{3,6})\s+NBM\s+\S+\s+NBP\s+GUIDANCE\s+"
                     r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{2})(\d{2})\s+UTC")


def _parse_block(lines: list[str], station: str) -> dict[str, dict] | None:
    """One station block → {local 'YYYY-MM-DD': {p10..p90, mean, sd}} (MaxT °F)."""
    m = _HDR_RE.match(lines[0])
    if not m or m.group(1) != station:
        return None
    # Bulletin header time is UTC by definition (".. 0100 UTC"); same localize
    # pattern as ingest/kalshi.py for source-declared-UTC stamps.
    cyc_dt = pd.Timestamp(datetime(int(m.group(4)), int(m.group(2)),
                                   int(m.group(3)), int(m.group(5)),
                                   int(m.group(6)))).tz_localize("UTC")
    utc_row = next((l for l in lines if l.startswith(" UTC")), None)
    fhr_row = next((l for l in lines if l.startswith(" FHR")), None)
    if not utc_row or not fhr_row:
        return None
    # Column character-spans from the FHR row; every element row is
    # right-aligned to the same column ends, 3-char fields.
    spans = [(mm.start(), mm.end()) for mm in re.finditer(r"\d+", fhr_row)]

    def _vals(row: str) -> list[int | None]:
        out = []
        for _, e in spans:
            tok = row[max(0, e - 3):e].strip()
            out.append(int(tok) if tok.lstrip("-").isdigit() else None)
        return out

    utc_vals = _vals(utc_row)
    fhr_vals = _vals(fhr_row)

    from kalshi_weather.tz import lst_offset
    try:
        offset = lst_offset(station)
    except Exception:
        return None

    out: dict[str, dict] = {}
    rows = {l[:6].strip(): l for l in lines[1:] if l[:6].strip() in _PCT_ROWS}
    for i, (utc_h, fhr) in enumerate(zip(utc_vals, fhr_vals)):
        if utc_h != 0 or fhr is None:
            continue                       # MaxT lives in the 00Z columns only
        valid = cyc_dt + timedelta(hours=int(fhr))
        # round to the exact 00Z instant (FHR from odd cycles lands at :00 already)
        valid = valid.replace(minute=0)
        local_day = (valid + offset).date().isoformat()
        q: dict[str, int] = {}
        for label, key in _PCT_ROWS.items():
            r = rows.get(label)
            if r is not None:
                v = _vals(r)[i]
                if v is not None:
                    q[key] = v
        if all(k in q for k in ("p10", "p25", "p50", "p75", "p90")):
            out[local_day] = q
    return out or None


def _extract_stations(text_iter, stations: set[str]) -> dict[str, dict]:
    """Stream lines once; parse only the blocks belonging to `stations`."""
    found: dict[str, dict] = {}
    block: list[str] = []
    cur: str | None = None
    for raw in text_iter:
        line = raw.rstrip("\n")
        h = _HDR_RE.match(line)
        if h:
            if cur and block:
                parsed = _parse_block(block, cur)
                if parsed:
                    found[cur] = parsed
                if len(found) == len(stations):
                    return found          # all wanted blocks parsed — stop streaming
            sid = h.group(1)
            cur = sid if sid in stations else None
            block = [line] if cur else []
        elif cur:
            block.append(line)
    if cur and block:
        parsed = _parse_block(block, cur)
        if parsed:
            found[cur] = parsed
    return found


def _cache_path(ymd: str, cyc: int) -> Path:
    return CACHE_DIR / f"nbp_{ymd}_{cyc:02d}z.json"


def fetch_cycle(ymd: str, cyc: int, stations: set[str]) -> dict[str, dict] | None:
    """Parsed MaxT quantiles for one cycle: cache hit, else stream from NOMADS."""
    cp = _cache_path(ymd, cyc)
    if cp.exists():
        try:
            doc = json.loads(cp.read_text())
            if set(stations) <= set(doc.get("stations", {})) or doc.get("complete"):
                return doc["stations"]
        except (OSError, json.JSONDecodeError):
            pass
    import requests
    try:
        resp = requests.get(NBM_URL.format(ymd=ymd, cyc=cyc), stream=True,
                            timeout=(10, 180))
        if resp.status_code != 200:
            return None
        found = _extract_stations(
            (l.decode("utf-8", "replace") for l in resp.iter_lines()), stations)
    except Exception:
        return None
    if not found:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"fetched_at": utc_now().isoformat(),
                              "complete": True, "stations": found}))
    return found


def latest_quantiles(station: str, settlement_date: str | date,
                     stations: set[str] | None = None,
                     now: datetime | None = None) -> dict | None:
    """Newest available NBP MaxT quantiles for (station, LOCAL settlement day).

    Walks full cycles newest-first within the lookback; one network fetch per
    cycle at most (then cached for every station on the same board run).
    """
    now = now or utc_now()
    sdate = str(settlement_date)
    stations = stations or {station}
    t = now - timedelta(hours=_AVAIL_LAG_H)
    for back in range(int(_LOOKBACK_H) + 1):
        dt = t - timedelta(hours=back)
        if dt.hour in FULL_CYCLES:
            got = fetch_cycle(dt.strftime("%Y%m%d"), dt.hour, stations)
            if got and station in got and sdate in got[station]:
                q = dict(got[station][sdate])
                q["cycle"] = f"{dt.strftime('%Y%m%d')}_{dt.hour:02d}Z"
                return q
    return None


def nbm_prob(strike_type: str, floor: float | None, cap: float | None,
             q: dict) -> float | None:
    """P(YES) under the NBM quantile distribution, Kalshi integer settlement.

    CDF: linear interpolation through the five known quantile points; normal
    tails anchored at the median with sigma = (p90 − p10) / 2.5631 (the exact
    normal 10–90 spread). Integer settlement → evaluate at ±0.5 boundaries.
    """
    try:
        pts = [(float(q["p10"]), 0.10), (float(q["p25"]), 0.25),
               (float(q["p50"]), 0.50), (float(q["p75"]), 0.75),
               (float(q["p90"]), 0.90)]
    except (KeyError, TypeError, ValueError):
        return None
    mu = pts[2][0]
    sigma = max((pts[4][0] - pts[0][0]) / 2.5631, 0.5)

    def cdf(x: float) -> float:
        if x <= pts[0][0] or x >= pts[4][0]:
            return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))
        for (x0, f0), (x1, f1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return f0 if x1 == x0 else f0 + (f1 - f0) * (x - x0) / (x1 - x0)
        return 0.5   # unreachable

    from kalshi_weather.settlement import yes_bounds
    b = yes_bounds(strike_type, floor, cap)
    if b is None:
        return None
    lo, hi = b
    lo_p = 0.0 if lo == -math.inf else cdf(lo)
    hi_p = 1.0 if hi == math.inf else cdf(hi)
    return round(hi_p - lo_p, 3)
