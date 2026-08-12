"""
Replay harness: re-derive logged model outputs through the CURRENT code and diff.

Every signals row logs its inputs at each transformation seam (prob_raw before
calibration, market mid before edge/direction). This re-runs those seams with the
code as it exists NOW and compares against what production actually emitted:

  Layer A — calibration: prob_raw → apply_prob_calibration(current map) must
            reproduce the logged prob_estimate. Compared ONLY for rows produced
            under the current calibration artifact (provenance calib_hash match;
            legacy rows without a stamp fall back to run_ts vs map mtime) — older
            rows differ legitimately and are reported as skipped, not drift.
  Layer B — decision: edge_raw must equal |prob_estimate − market_mid| and
            direction must match its sign. This is the seam where the 1°F
            boundary bug and a direction misread would show up.

Interpretation: DRIFT means the current code produces different numbers than
production did for identical inputs. After an INTENTIONAL model change that is
expected — run with --ack to acknowledge (the diff summary is still printed, so
the change is reviewed, not rubber-stamped). Unacknowledged drift exits 1 and
blocks a push.

Not covered (tier 2): re-pricing from raw ensemble members (needs the member
archive join + point-in-time bias), and same-day intraday recompute (obs inputs
are only partially logged). Layer A/B still covers every past pricing incident
that happened downstream of the distribution itself.

Usage:
  .venv/bin/python scripts/replay_check.py --days 3          # pre-push gate
  .venv/bin/python scripts/replay_check.py --days 7 --ack    # informational
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalshi_weather.calibration.recalibrate import (          # noqa: E402
    apply_prob_calibration, get_calibration, lead_bucket, _DEFAULT_PATH as MAP_PATH)
from kalshi_weather.provenance import provenance              # noqa: E402

SIGNALS = ROOT / "data" / "signals" / "signals_log.jsonl"

TOL_PROB = 2e-3     # logged values are rounded to 4dp; recompute from rounded
TOL_EDGE = 2e-3     # inputs can differ in the 4th decimal — never at bug scale


def load_rows(days: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = []
    try:
        with open(SIGNALS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("run_ts", "")) >= cutoff:
                    rows.append(r)
    except FileNotFoundError:
        pass
    return rows


def _calib_comparable(r: dict, current_hash: str | None, map_mtime: float) -> bool:
    """Only rows produced under the CURRENT calibration artifact replay exactly."""
    if r.get("calib_hash") is not None:
        return r["calib_hash"] == current_hash
    ts = str(r.get("run_ts", ""))
    try:
        return datetime.fromisoformat(ts).timestamp() > map_mtime   # legacy rows
    except ValueError:
        return False


def check_rows(rows: list[dict], maps: dict, current_hash: str | None,
               map_mtime: float) -> dict:
    out = {"n": len(rows), "calib_checked": 0, "calib_skipped_era": 0,
           "calib_drift": [], "edge_drift": [], "dir_drift": [], "missing": 0}
    for r in rows:
        raw, est = r.get("prob_raw"), r.get("prob_estimate")
        mid, direction = r.get("market_mid"), r.get("direction")
        if None in (raw, est, mid, direction) or r.get("prob_source") is None:
            out["missing"] += 1
            continue

        # Layer A — calibration seam
        if _calib_comparable(r, current_hash, map_mtime):
            out["calib_checked"] += 1
            expect = float(np.clip(float(raw), 1e-6, 1 - 1e-6))
            # Lead-keyed maps: replay with the lead production logged (calib_lead);
            # rows that predate the field derive it from their logged horizon.
            lead = r.get("calib_lead", lead_bucket(r.get("hours_to_settle")))
            expect = apply_prob_calibration(expect, r["prob_source"], maps, lead=lead)
            expect = float(np.clip(expect, 1e-6, 1 - 1e-6))
            if abs(expect - float(est)) > TOL_PROB:
                out["calib_drift"].append(
                    (r.get("ticker"), r.get("run_ts"),
                     f"logged {est:.4f} vs recomputed {expect:.4f} (raw {raw:.4f})"))
        else:
            out["calib_skipped_era"] += 1

        # Layer B — edge/direction seam (era-independent: pure arithmetic)
        signed = float(est) - float(mid)
        want_dir = "BUY_YES" if signed >= 0 else "BUY_NO"
        if direction != want_dir and abs(signed) > TOL_EDGE:   # ties can go either way
            out["dir_drift"].append(
                (r.get("ticker"), r.get("run_ts"),
                 f"logged {direction} but prob {est:.4f} vs mid {mid:.4f} says {want_dir}"))
        if abs(abs(signed) - float(r.get("edge_raw", 0.0))) > TOL_EDGE:
            out["edge_drift"].append(
                (r.get("ticker"), r.get("run_ts"),
                 f"logged edge {r.get('edge_raw')} vs |{est:.4f}-{mid:.4f}|={abs(signed):.4f}"))
    return out


def report(days: int) -> tuple[str, bool]:
    rows = load_rows(days)
    if not rows:
        return f"replay: no signals rows in the last {days}d — nothing to compare", False
    try:
        map_mtime = MAP_PATH.stat().st_mtime
    except OSError:
        map_mtime = 0.0
    res = check_rows(rows, get_calibration(), provenance().get("calib_hash"), map_mtime)

    drift = res["calib_drift"] + res["edge_drift"] + res["dir_drift"]
    lines = [f"REPLAY ({days}d): {res['n']} rows | calib replayed {res['calib_checked']} "
             f"(skipped {res['calib_skipped_era']} older-artifact) | "
             f"missing-fields {res['missing']}"]
    for name, items in (("CALIBRATION", res["calib_drift"]),
                        ("EDGE", res["edge_drift"]), ("DIRECTION", res["dir_drift"])):
        if items:
            lines.append(f"  🔴 {name} drift × {len(items)} — first 5:")
            lines += [f"     {t} @ {ts}: {msg}" for t, ts, msg in items[:5]]
    if not drift:
        lines.append("  ✅ current code reproduces production output exactly")
    return "\n".join(lines), bool(drift)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--ack", action="store_true",
                    help="acknowledge INTENTIONAL model-behavior drift (still prints the diff)")
    args = ap.parse_args()
    text, drift = report(args.days)
    print(text)
    if drift and not args.ack:
        print("\n🔴 UNACKNOWLEDGED MODEL DRIFT — current code disagrees with production "
              "on identical inputs.\nIf intentional, re-run/push with REPLAY_ACK "
              "(--ack); otherwise this is a regression. Do NOT land.")
        sys.exit(1)
