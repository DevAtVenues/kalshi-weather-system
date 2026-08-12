"""
Model skill — which forecast model actually wins, per city, and does a blend beat them?

Joins the multi-model archive (archive_forecasts.py) to the actual CLI high and reports, per
station, each model's mean-absolute error + bias at a fixed lead, ranks them, and checks
whether a simple average beats the best single model. This is what tells us how to WEIGHT the
models per city instead of the fixed blend we use now — and it uses CONTINUOUS error (high
information per day), not binary win/loss.

Usage: .venv/bin/python scripts/model_skill.py [--lead 1]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kalshi_weather.ingest.labels import fetch_labels  # noqa: E402

ARCHIVE = ROOT / "data" / "logger" / "forecasts" / "forecasts.jsonl"
MODELS = ["gfs", "ecmwf", "nbm", "hrrr", "nws"]


def _actuals(stations_years: set[tuple[str, int]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    cur = date.today().year
    by_stn: dict[str, set[int]] = defaultdict(set)
    for stn, yr in stations_years:
        by_stn[stn].add(yr)
    for stn, yrs in by_stn.items():
        for yr in yrs:
            try:
                df = fetch_labels(stn, [yr], force_refresh=(yr == cur))
            except Exception:
                continue
            for _, row in df.iterrows():
                h = row.get("high")
                if h is not None and h == h:          # skip None and NaN
                    out[(stn, str(row["date"]))] = float(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=1, help="lead days to evaluate (1 = day-ahead)")
    args = ap.parse_args()

    files = [f for f in (ARCHIVE, ARCHIVE.with_name("forecasts_backfill.jsonl")) if f.exists()]
    if not files:
        print("No forecast archive yet — run scripts/archive_forecasts.py, or "
              "scripts/backfill_forecasts.py for history now.")
        return
    # dedup to the LAST snapshot per (station, settlement_date) at the requested lead.
    # Forward snapshots (real ISO ts) beat backfill rows (ts "0000-…") on any overlap.
    latest: dict[tuple, dict] = {}
    for f in files:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("lead_days") != args.lead:
                continue
            k = (r["station"], r["settlement_date"])
            if k not in latest or r["snapshot_ts"] > latest[k]["snapshot_ts"]:
                latest[k] = r

    today = date.today().isoformat()
    settled = [r for r in latest.values() if r["settlement_date"] < today]
    if not settled:
        print(f"Archive has {len(latest)} rows at lead={args.lead}, but none settled yet — "
              "check back after they settle + grade.")
        return

    # Legacy rows archived "gfs" via gfs_seamless, which IS HRRR for the first ~2 days —
    # their gfs column duplicates hrrr and must not count as GFS skill. Rows written after
    # the fix carry gfs_model="gfs_global"; anything else gets its gfs reading dropped.
    dropped = 0
    for r in settled:
        if r.get("gfs") is not None and r.get("gfs_model") != "gfs_global":
            r["gfs"] = None
            dropped += 1
    if dropped:
        print(f"  (excluded {dropped} legacy rows whose 'gfs' was gfs_seamless ≈ HRRR)")

    sy = {(r["station"], int(r["settlement_date"][:4])) for r in settled}
    actual = _actuals(sy)

    # per station per model: absolute errors + signed errors
    err: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    blend_err: dict[str, list[float]] = defaultdict(list)
    days = set()
    for r in settled:
        a = actual.get((r["station"], r["settlement_date"]))
        if a is None:
            continue
        days.add(r["settlement_date"])
        preds = [r[m] for m in MODELS if r.get(m) is not None]
        if preds:
            blend_err[r["station"]].append(a - sum(preds) / len(preds))
        for m in MODELS:
            if r.get(m) is not None:
                err[r["station"]][m].append(a - r[m])

    print("=" * 72)
    print(f"  MODEL SKILL — lead {args.lead}d, {len({(r['station'],r['settlement_date']) for r in settled})} "
          f"city-days over {len(days)} distinct dates (need many more to trust)")
    print("=" * 72)
    print(f"  {'CITY':<5}" + "".join(f"{m.upper():>10}" for m in MODELS) + f"{'BLEND':>10}{'  best':>8}")
    pooled = defaultdict(list); pooled_blend = []
    for stn in sorted(err):
        cells = []
        maes = {}
        for m in MODELS:
            e = err[stn].get(m, [])
            if e:
                mae = st.mean(abs(x) for x in e); maes[m] = mae
                cells.append(f"{mae:>10.2f}"); pooled[m].extend(e)
            else:
                cells.append(f"{'—':>10}")
        be = blend_err.get(stn, [])
        bmae = st.mean(abs(x) for x in be) if be else None
        if be: pooled_blend.extend(be)
        best = min(maes, key=maes.get) if maes else "—"
        print(f"  {stn[1:]:<5}" + "".join(cells) + (f"{bmae:>10.2f}" if bmae else f"{'—':>10}") + f"{best.upper():>8}")
    # pooled MAE + bias
    print("  " + "-" * 70)
    line = f"  {'ALL':<5}"
    for m in MODELS:
        line += f"{st.mean(abs(x) for x in pooled[m]):>10.2f}" if pooled[m] else f"{'—':>10}"
    line += f"{st.mean(abs(x) for x in pooled_blend):>10.2f}" if pooled_blend else f"{'—':>10}"
    best_pooled = min((m for m in MODELS if pooled[m]), key=lambda m: st.mean(abs(x) for x in pooled[m]), default="—")
    print(line + f"{best_pooled.upper():>8}  (MAE)")
    bias = "  bias:" + "".join(f"  {m}={st.mean(pooled[m]):+.1f}" for m in MODELS if pooled[m])
    print(bias + "   (+ = model runs cool / actual hotter)")


if __name__ == "__main__":
    main()
