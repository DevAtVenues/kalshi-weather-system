#!/usr/bin/env python3
"""
The corrected ensemble's real forecast accuracy — the number the docs never produced.

docs/STATE_OF_THE_MODEL.md leads with "MAE 1.52F" as proof the forecast is skilled.
That number comes from model_skill_report.py:95, which differences `actual_temp`
against `tmax_f_fcst` — and `tmax_f_fcst` is populated at runner.py:588 from
fetch_gfs_tmax(), the raw single-run GFS. It is not the corrected ensemble that
prices contracts; that centre is a different logged field, `ens_p50`. So the
headline skill figure describes a forecast the trading engine does not use
(audit F16). The ensemble's own accuracy is nowhere in the reports.

Both fields are already logged per graded outcome, so this is a query, not a
project. It reports:

  * corrected-ensemble MAE and bias, with a day-clustered bootstrap CI
  * raw-GFS MAE on the SAME rows — the apples-to-apples the docs never ran
  * both split by forecast lead, because trades only fire inside 36h
  * per-station bias, to size the coastal settlement-sensor problem

Two deliberate methodology choices, both departures from this repo:

1. Rows priced after the settlement window opened are DROPPED. On the settlement
   day the members are conditioned on the observed running high (runner.py:515),
   so `ens_p50` there is part observation and would flatter the result. Only
   pre-window forecasts are honest. --min-lead-h changes the cut.

2. The bootstrap resamples SETTLEMENT DAYS with replacement, carrying all of a
   day's cities together. backtest.py:269 instead takes `n_dates // 30` blocks,
   which is a single block below 60 days and returns a ZERO-WIDTH interval when
   the day count divides 30 (audit F31). Do not reuse that function.

Usage
  python scripts/ensemble_mae.py
  python scripts/ensemble_mae.py --min-lead-h 14 --station KNYC
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"

# Lead buckets in hours. Picks can only fire inside HORIZON_NEVER_FIRE_H = 36,
# so "0-14" and "14-40" are the buckets that describe tradeable decisions.
BUCKETS = [("same-day 0-14h", 0.0, 14.0),
           ("day-ahead 14-40h", 14.0, 40.0),
           ("long 40h+", 40.0, float("inf"))]


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No graded outcomes at {path}.\n"
            "  This file is written by outcome_tracker.grade_signals(), which runs\n"
            "  from scripts/daily_review.py. Without it there is nothing to measure —\n"
            "  the inputs live only on the machine that ran the engine.")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def usable(rows: list[dict], min_lead_h: float, station: str | None) -> list[dict]:
    """Ensemble-priced, graded, pre-settlement-window rows with both centres."""
    out = []
    for r in rows:
        if r.get("prob_source") != "ensemble":
            continue
        if station and r.get("station") != station:
            continue
        actual, p50 = r.get("actual_temp"), r.get("ens_p50")
        if actual is None or p50 is None:
            continue
        lead = r.get("hours_to_settle")
        if lead is None or not isinstance(lead, (int, float)) or lead != lead:
            continue                      # NaN or missing lead: cannot place it
        if lead < min_lead_h:
            continue                      # window open / too near: p50 is conditioned
        out.append(r)
    return out


def per_city_day(rows: list[dict]) -> list[dict]:
    """One observation per (city, settlement_date) — a city-day is the unit of
    forecast skill. run_live logs the same contract every 30 min; counting each
    snapshot would inflate n by the number of cycles a pick survived (audit F9).
    Keep the LAST pre-window snapshot: the forecast the decision actually stood on."""
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r.get("city"), r.get("settlement_date"))
        cur = best.get(k)
        if cur is None or str(r.get("run_ts", "")) > str(cur.get("run_ts", "")):
            best[k] = r
    return list(best.values())


def errors(rows: list[dict]) -> list[dict]:
    """Signed error (actual − forecast) for both centres, on identical rows."""
    out = []
    for r in rows:
        actual = float(r["actual_temp"])
        ens = actual - float(r["ens_p50"])
        gfs = (actual - float(r["tmax_f_fcst"])
               if r.get("tmax_f_fcst") is not None else None)
        out.append({"city": r.get("city"), "station": r.get("station"),
                    "day": str(r.get("settlement_date")), "lead": float(r["hours_to_settle"]),
                    "ens_err": ens, "gfs_err": gfs})
    return out


def day_bootstrap(errs: list[dict], field: str, stat, n_boot: int = 5000,
                  seed: int = 0) -> tuple[float, float, float]:
    """Cluster bootstrap by settlement day: resample DAYS with replacement, each
    carrying all of its city rows, so weather autocorrelation within a day is
    respected. Returns (point, lo, hi) at 95%."""
    import random
    by_day: dict[str, list[float]] = defaultdict(list)
    for e in errs:
        v = e.get(field)
        if v is not None:
            by_day[e["day"]].append(v)
    days = list(by_day)
    flat = [v for d in days for v in by_day[d]]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = stat(flat)
    if len(days) < 2:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        picked = [rng.choice(days) for _ in range(len(days))]
        vals = [v for d in picked for v in by_day[d]]
        if vals:
            means.append(stat(vals))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return point, lo, hi


def mae(vals):  return sum(abs(v) for v in vals) / len(vals)
def bias(vals): return sum(vals) / len(vals)


def report(errs: list[dict], label: str) -> list[str]:
    L = []
    n_days = len({e["day"] for e in errs})
    paired = [e for e in errs if e["gfs_err"] is not None]
    if not errs:
        return [f"  {label:<20} (no rows)"]
    e_mae, e_lo, e_hi = day_bootstrap(errs, "ens_err", mae)
    e_b, eb_lo, eb_hi = day_bootstrap(errs, "ens_err", bias)
    L.append(f"  {label:<20} n={len(errs):>4} days={n_days:>3}   "
             f"ENSEMBLE MAE {e_mae:.2f}F [{e_lo:.2f}, {e_hi:.2f}]   "
             f"bias {e_b:+.2f}F [{eb_lo:+.2f}, {eb_hi:+.2f}]")
    if paired:
        g_mae, g_lo, g_hi = day_bootstrap(paired, "gfs_err", mae)
        diffs = [{"day": e["day"], "d": abs(e["gfs_err"]) - abs(e["ens_err"])}
                 for e in paired]
        d_pt, d_lo, d_hi = day_bootstrap(
            [{"day": d["day"], "d": d["d"]} for d in diffs], "d", bias)
        verdict = ("ensemble BETTER" if d_lo > 0 else
                   "raw GFS BETTER" if d_hi < 0 else "not distinguishable")
        L.append(f"  {'':<20} {'':>10}           "
                 f"raw GFS  MAE {g_mae:.2f}F [{g_lo:.2f}, {g_hi:.2f}]")
        L.append(f"  {'':<20} {'':>10}           "
                 f"improvement {d_pt:+.2f}F [{d_lo:+.2f}, {d_hi:+.2f}] — {verdict}")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcomes", type=Path, default=OUTCOMES)
    ap.add_argument("--min-lead-h", type=float, default=0.0,
                    help="drop rows priced with less lead than this (default 0 = "
                         "strictly before the settlement window opens)")
    ap.add_argument("--station", help="restrict to one settlement station, e.g. KNYC")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    raw = load(args.outcomes)
    keep = usable(raw, args.min_lead_h, args.station)
    rows = per_city_day(keep)
    errs = errors(rows)

    print("=" * 78)
    print("  CORRECTED-ENSEMBLE FORECAST SKILL")
    print("=" * 78)
    print(f"\n  {len(raw)} graded outcomes → {len(keep)} ensemble-priced pre-window rows"
          f" → {len(rows)} city-days")
    if args.min_lead_h <= 0:
        print("  (rows priced after the settlement window opened are excluded: their")
        print("   ens_p50 is conditioned on the observed running high)")
    if not errs:
        print("\n  Nothing to measure. Either no ensemble-priced picks have settled,")
        print("  or ens_p50 was never populated (check prob_source in the outcomes).")
        return 1

    print()
    for line in report(errs, "ALL LEADS"):
        print(line)
    print()
    for name, lo, hi in BUCKETS:
        sub = [e for e in errs if lo <= e["lead"] < hi]
        if sub:
            for line in report(sub, name):
                print(line)

    by_station: dict[str, list[dict]] = defaultdict(list)
    for e in errs:
        by_station[e["station"] or "?"].append(e)
    if len(by_station) > 1:
        print(f"\n  per-station signed bias (actual − ensemble centre; + = we forecast COOL)")
        print(f"  {'stn':<6} {'n':>4} {'days':>5} {'bias':>8} {'MAE':>7}")
        for stn, es in sorted(by_station.items(),
                              key=lambda kv: -abs(bias([e['ens_err'] for e in kv[1]]))):
            v = [e["ens_err"] for e in es]
            flag = "  ← investigate" if abs(bias(v)) >= 1.0 else ""
            print(f"  {stn:<6} {len(es):>4} {len({e['day'] for e in es}):>5} "
                  f"{bias(v):>+8.2f} {mae(v):>7.2f}{flag}")

    print("\n  How to read this:")
    print("   · If the ensemble's CI does not clear the raw-GFS CI, the corrections")
    print("     in calibration/ are not earning their complexity.")
    print("   · A per-station bias above ~1F is uncorrected settlement-sensor error;")
    print("     audit F23 shows the centre-bias controller only ever removes ~47% of it.")
    print("   · This measures the FORECAST only. It says nothing about whether the")
    print("     forecast beats the market price — that is reconcile_pnl.py's job.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "n_city_days": len(rows), "min_lead_h": args.min_lead_h,
            "errors": errs}, indent=1))
        print(f"\n  → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
