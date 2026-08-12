"""
Fit the market-shrinkage weight w per {source}@{lead} on graded outcomes.

    p_post = sigmoid( w * logit(p_model) + (1 - w) * logit(p_market) )

Population per key (mirrors the calibration-map / skill_monitor discipline):
graded rows of `source` whose hours_to_settle falls in the lead bucket,
deduped to the LAST read per ticker (cycle re-logs of one contract are one
bet), degenerate quotes (mid outside [0.005, 0.995]) dropped.

Fit: grid search w ∈ [0, 1] step 0.05 minimizing mean log-loss. Evidence:
leave-one-DAY-out (days, not rows — same-day outcomes are one weather
system): per held-out day the weight is refit on the remaining days, and the
pooled OOS log-loss/Brier is reported for market-only (w=0), model-only
(w=1) and the fitted blend, plus the per-fold w spread (stability). Keys
with insufficient sample are skipped, never defaulted.

The artifact is consumed LOG-ONLY by the runner (prob_shrunk / edge_shrunk);
nothing gates or sizes on it until its forward record supports promotion.

Usage:  .venv/bin/python scripts/build_shrinkage.py [--sources ensemble]
Output: data/models/prob_shrinkage.json + stdout report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalshi_weather.calibration.recalibrate import lead_bucket   # noqa: E402
from kalshi_weather.calibration.shrinkage import blend           # noqa: E402

OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"
OUT      = ROOT / "data" / "models" / "prob_shrinkage.json"

MIN_N    = 100
MIN_DAYS = 7
W_GRID   = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def load_population(source: str, lead: str) -> list[dict]:
    last: dict[str, dict] = {}
    rows = []
    with open(OUTCOMES) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for r in sorted(rows, key=lambda r: str(r.get("run_ts", ""))):
        if r.get("prob_source") != source or r.get("yes_settled") is None:
            continue
        if lead_bucket(r.get("hours_to_settle")) != lead:
            continue
        pm, mid = r.get("prob_estimate"), r.get("market_mid")
        if pm is None or mid is None:
            continue
        mid = float(mid)
        if not (0.005 <= mid <= 0.995):
            continue                      # settled/degenerate quote — not a market view
        last[str(r.get("ticker"))] = {
            "x": float(pm), "m": mid,
            "y": 1.0 if r["yes_settled"] else 0.0,
            "d": str(r.get("settlement_date", "")),
        }
    return sorted(last.values(), key=lambda r: r["d"])


def fit_w(pop: list[dict]) -> float:
    x = np.array([r["x"] for r in pop])
    m = np.array([r["m"] for r in pop])
    y = np.array([r["y"] for r in pop])
    losses = [_logloss(np.array([blend(xi, mi, w) for xi, mi in zip(x, m)]), y)
              for w in W_GRID]
    return float(W_GRID[int(np.argmin(losses))])


def lodo(pop: list[dict]) -> dict:
    days = sorted({r["d"] for r in pop})
    preds_fit, preds_mkt, preds_mod, ys, fold_ws = [], [], [], [], []
    for hold in days:
        tr = [r for r in pop if r["d"] != hold]
        te = [r for r in pop if r["d"] == hold]
        if not tr or not te:
            continue
        w = fit_w(tr)
        fold_ws.append(w)
        for r in te:
            preds_fit.append(blend(r["x"], r["m"], w))
            preds_mkt.append(r["m"])
            preds_mod.append(r["x"])
            ys.append(r["y"])
    y = np.array(ys)
    return {
        "days": len(days), "n": len(ys),
        "lodo_logloss": {"market": round(_logloss(np.array(preds_mkt), y), 4),
                         "model": round(_logloss(np.array(preds_mod), y), 4),
                         "blend": round(_logloss(np.array(preds_fit), y), 4)},
        "lodo_brier":   {"market": round(_brier(np.array(preds_mkt), y), 4),
                         "model": round(_brier(np.array(preds_mod), y), 4),
                         "blend": round(_brier(np.array(preds_fit), y), 4)},
        "fold_w_min": min(fold_ws), "fold_w_max": max(fold_ws),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["ensemble"])
    args = ap.parse_args()

    doc = {"meta": {"fit_at": datetime.now(tz=timezone.utc).isoformat(),
                    "note": "LOG-ONLY consumer (prob_shrunk/edge_shrunk); "
                            "promotion to gating requires forward evidence",
                    "grid_step": 0.05, "min_n": MIN_N, "min_days": MIN_DAYS,
                    "skipped": []},
           "weights": {}}
    for source in args.sources:
        for lead in ("same-day", "day-ahead"):
            key = f"{source}@{lead}"
            pop = load_population(source, lead)
            days = len({r["d"] for r in pop})
            if len(pop) < MIN_N or days < MIN_DAYS:
                doc["meta"]["skipped"].append(
                    f"{key}: insufficient (n={len(pop)}, days={days})")
                print(f"{key}: SKIPPED (n={len(pop)}, days={days})")
                continue
            w = fit_w(pop)
            ev = lodo(pop)
            doc["weights"][key] = {"w": w, "n": len(pop), "days": days,
                                   "evidence": ev}
            ll = ev["lodo_logloss"]
            print(f"{key}: w={w:.2f} (n={len(pop)}, days={days}) | LODO logloss "
                  f"market {ll['market']:.4f} / model {ll['model']:.4f} / "
                  f"blend {ll['blend']:.4f} | fold w [{ev['fold_w_min']:.2f},"
                  f"{ev['fold_w_max']:.2f}]")

    OUT.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
