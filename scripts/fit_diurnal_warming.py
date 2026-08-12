"""
Fit the intraday warming curve β(h) from real hourly obs + archived forecasts.

WHY: intraday_prob's original taper was linear in clock time (sunrise 7 → peak 15),
i.e. it assumed ~62% of the day's warming is done by noon. Checked against live obs
2026-07-12 that is badly wrong (July highs peak 4-6 PM local; MSP was at the model's
projected FINAL high at 1 PM with hours of climb left). The taper multiplies the
member forecast's upside above the running high, so a too-small fraction drags the
distribution center below the forecast and biases every same-day P(YES) toward
"the high stays near the running high".

THE FIT: for each (station, day) pair join
  fcst  = day-ahead HRRR tmax (data/logger/forecasts/forecasts_backfill.jsonl +
          live archive lead<=1; HRRR is the measured-best model, MAE 1.22°F)
  RH(h) = running high through LST hour h (IEM hourly METARs)
  final = the day's max hourly temp
and regress through the origin, per LST hour h:
  (final - RH(h)) = β(h) · (fcst - RH(h)) + ε
β(h) is exactly the multiplier intraday_adjust needs: how much of the forecast's
remaining upside is realized on average, GIVEN the running high at hour h. If the
forecast were a perfect predictor of the final high β would be 1 at every hour; β<1
reflects both forecast over-shoot and the information in a lagging running high.

Output: data/calibration/diurnal_warming_curve.json
  {"beta": {hour: β}, "resid_sd": {hour: sd of ε}, "n": {hour: samples}, ...}
resid_sd doubles as the empirical width for adj_sigma at that hour.

NOTE the label here is the IEM hourly max, not the CLI settlement value — fine for
fitting the *shape* of the warming curve (same-station, same-day comparison; the
CLI/METAR gap is a level offset handled elsewhere).

Usage:
  .venv/bin/python scripts/fit_diurnal_warming.py             # fit + write
  .venv/bin/python scripts/fit_diurnal_warming.py --start 2026-05-15 --end 2026-07-11
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalshi_weather.tz import lst_offset  # noqa: E402

FORECASTS = [
    ROOT / "data" / "logger" / "forecasts" / "forecasts_backfill.jsonl",
    ROOT / "data" / "logger" / "forecasts" / "forecasts.jsonl",
]
OUT = ROOT / "data" / "calibration" / "diurnal_warming_curve.json"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
HOURS = list(range(7, 17))          # LST hours to fit (7 AM .. 4 PM)
MIN_UPSIDE_F = 0.5                  # skip samples where fcst has ~no upside left (ill-conditioned)
MIN_OBS_PER_DAY = 16                # require decent hourly coverage for a day to count


def load_forecasts() -> pd.DataFrame:
    """Day-ahead (lead<=1) tmax forecast per (station, settlement_date); HRRR first."""
    rows = []
    for path in FORECASTS:
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("lead_days") not in (0, 1):
                    continue
                fc = next((r[m] for m in ("hrrr", "ecmwf", "nbm", "gfs")
                           if r.get(m) is not None), None)
                if fc is None:
                    continue
                rows.append({"station": r["station"], "date": r["settlement_date"],
                             "fcst": float(fc), "lead": r["lead_days"]})
    df = pd.DataFrame(rows)
    # Prefer the shortest lead per (station, date) when both archives cover a day.
    df = df.sort_values("lead").drop_duplicates(["station", "date"], keep="first")
    return df


def fetch_iem_hourly(station: str, start: str, end: str) -> pd.DataFrame:
    """Hourly METAR temps (°F) from IEM ASOS, UTC timestamps."""
    params = {
        "station": station.lstrip("K"), "data": "tmpf",
        "year1": start[:4], "month1": int(start[5:7]), "day1": int(start[8:10]),
        "year2": end[:4], "month2": int(end[5:7]), "day2": int(end[8:10]),
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
        "missing": "empty", "trace": "empty", "report_type": 3,
    }
    url = f"{IEM}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "plec-diurnal-fit"})
    raw = None
    for attempt in range(5):                      # IEM rate-limits bursts: back off and retry
        try:
            raw = urllib.request.urlopen(req, timeout=60).read().decode()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    time.sleep(5)                                 # pace between stations
    df = pd.read_csv(io.StringIO(raw))
    if df.empty or "tmpf" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["tmpf"])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df[["valid", "tmpf"]]


def day_curves(obs: pd.DataFrame, station: str) -> pd.DataFrame:
    """Per (LST day, LST hour): running high so far + the day's final hourly max."""
    off = lst_offset(station)
    t = obs.copy()
    t["lst"] = t["valid"] + off
    t["day"] = t["lst"].dt.date.astype(str)
    t["hour"] = t["lst"].dt.hour
    out = []
    for day, g in t.groupby("day"):
        if len(g) < MIN_OBS_PER_DAY:
            continue
        g = g.sort_values("lst")
        final = g["tmpf"].max()
        for h in HOURS:
            seen = g[g["hour"] <= h]["tmpf"]
            if seen.empty:
                continue
            out.append({"station": station, "date": day, "hour": h,
                        "rh": float(seen.max()), "final": float(final)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-15")
    ap.add_argument("--end", default="2026-07-11")
    args = ap.parse_args()

    fc = load_forecasts()
    fc = fc[(fc["date"] >= args.start) & (fc["date"] <= args.end)]
    stations = sorted(fc["station"].unique())
    print(f"forecasts: {len(fc)} station-days across {len(stations)} stations "
          f"({args.start}..{args.end})")

    curves = []
    for st in stations:
        try:
            obs = fetch_iem_hourly(st, args.start, args.end)
            dc = day_curves(obs, st)
            curves.append(dc)
            print(f"  {st}: {dc['date'].nunique() if not dc.empty else 0} usable days")
        except Exception as e:  # one bad station must not sink the fit
            print(f"  {st}: FETCH FAILED ({e})")
    allc = pd.concat(curves, ignore_index=True)
    m = allc.merge(fc, on=["station", "date"], how="inner")
    print(f"joined samples: {len(m)} hour-rows, {m.groupby(['station','date']).ngroups} station-days")

    beta, resid_sd, n_h, naive = {}, {}, {}, {}
    for h in HOURS:
        g = m[m["hour"] == h].copy()
        g["x"] = g["fcst"] - g["rh"]
        g["y"] = g["final"] - g["rh"]
        g = g[g["x"] >= MIN_UPSIDE_F]
        if len(g) < 30:
            continue
        b = float((g["x"] * g["y"]).sum() / (g["x"] ** 2).sum())
        beta[str(h)] = round(b, 4)
        resid_sd[str(h)] = round(float((g["y"] - b * g["x"]).std()), 4)
        n_h[str(h)] = int(len(g))
        naive[str(h)] = round(max(0.0, (15 - h) / 8.0), 4)   # old linear taper for comparison

    out = {
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
        "window": [args.start, args.end],
        "forecast_source": "hrrr (fallback ecmwf/nbm/gfs), lead<=1, backfill+live archive",
        "n_station_days": int(m.groupby(["station", "date"]).ngroups),
        "beta": beta,
        "resid_sd": resid_sd,
        "n": n_h,
        "old_linear_taper": naive,
        "note": "beta[h] multiplies (member - running_high) upside at LST hour h; "
                "resid_sd[h] is the empirical remaining-uncertainty width at that hour.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    print(f"{'h':>3} {'beta':>7} {'old':>6} {'resid_sd':>8} {'n':>5}")
    for h in HOURS:
        k = str(h)
        if k in beta:
            print(f"{h:>3} {beta[k]:>7.3f} {naive[k]:>6.3f} {resid_sd[k]:>8.3f} {n_h[k]:>5}")


if __name__ == "__main__":
    main()
