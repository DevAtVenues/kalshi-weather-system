"""
Model-skill report — separates the two questions the forward-edge report conflated:

  Q1 FORECAST SKILL: does the bias-corrected forecast predict the settled high?
     Measured per city-day → 20 cities give real cross-sectional sample even though
     only ~13 calendar days have accrued. Pure measurement, no overfitting risk.

  Q2 PROBABILITY EDGE: is our P(YES) better calibrated than the market's price?
     Brier(model) vs Brier(market) on the SAME contracts = the charter's literal
     definition of edge, across every graded contract (not just the 3 actionable days).

  Q3 THE DIAGNOSIS: if the forecast is skilled but reliability is overconfident, the
     bug is the VARIANCE (sigma too tight), not the forecast. We estimate the spread
     the residuals actually justify (leave-one-DAY-out, so a contract never sees its
     own day) and re-score Brier with it — labeled as an in-sample diagnostic that
     must be forward-validated, NOT a number to present as achieved.

All confidence intervals are BLOCK bootstraps over settlement DAYS (weather days are
autocorrelated; a heat wave is one shared outcome), and effective n is reported in
DAYS, not contracts — per the project's hard rules.

Usage:  .venv/bin/python scripts/model_skill_report.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from kalshi_weather.calibration.ensemble_dist import nws_normal_prob

_OUTCOMES = Path(__file__).parents[1] / "data" / "outcomes" / "signal_outcomes.jsonl"
_MIN_BLOCKS = 10


def _load() -> list[dict]:
    rows = [json.loads(l) for l in _OUTCOMES.read_text().splitlines() if l.strip()]
    # one row per ticker = its latest snapshot (the decision that stood)
    by_t: dict[str, dict] = {}
    for r in rows:
        k = r.get("ticker")
        if k and (k not in by_t or str(r.get("run_ts", "")) > str(by_t[k].get("run_ts", ""))):
            by_t[k] = r
    return list(by_t.values())


def _block_ci(values_by_day: dict[str, list[float]], n_boot: int = 5000, seed: int = 0):
    """Block bootstrap the mean, resampling whole DAYS. Returns (point, lo, hi, n_days)."""
    days = list(values_by_day.values())
    if not days:
        return None
    allv = [v for d in days for v in d]
    point = sum(allv) / len(allv)
    rng = random.Random(seed)
    k = len(days)
    means = []
    for _ in range(n_boot):
        s = []
        for _ in range(k):
            s.extend(days[rng.randrange(k)])
        if s:
            means.append(sum(s) / len(s))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return point, lo, hi, k


def _fmt_ci(ci, unit="", pct=False):
    if not ci:
        return "n/a"
    p, lo, hi, k = ci
    if pct:
        return f"{p:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  ({k} days)"
    return f"{p:.2f}{unit}  95% CI [{lo:.2f}, {hi:.2f}]{unit}  ({k} days)"


def main() -> None:
    rows = _load()
    fc = [r for r in rows
          if r.get("tmax_f_fcst") is not None and r.get("actual_temp") is not None]
    days = sorted({r["settlement_date"] for r in fc})
    print(f"\n{'='*70}\n  MODEL-SKILL REPORT — {len(rows)} contracts, {len(days)} calendar days")
    print(f"  (breadth is cross-sectional: {len({r['city'] for r in fc})} cities; the "
          f"time axis is the limit)\n{'='*70}")

    # ── Q1  FORECAST SKILL ────────────────────────────────────────────
    # De-dupe to one forecast error per city-day (a city-day is the unit of forecast skill).
    cd: dict[tuple, float] = {}
    for r in fc:
        err = float(r["actual_temp"]) - float(r["tmax_f_fcst"])   # +=actual warmer=model ran COOL
        cd[(r["city"], r["settlement_date"])] = err
    abs_by_day: dict[str, list[float]] = defaultdict(list)
    err_by_day: dict[str, list[float]] = defaultdict(list)
    for (city, day), e in cd.items():
        abs_by_day[day].append(abs(e))
        err_by_day[day].append(e)
    mae_ci = _block_ci(abs_by_day)
    bias_ci = _block_ci(err_by_day)
    print("Q1  FORECAST SKILL  (settled high − our corrected forecast, per city-day)")
    print(f"    unit = {len(cd)} city-days over {len(days)} days")
    print(f"    MAE        = {_fmt_ci(mae_ci, '°F')}")
    print(f"    mean bias  = {_fmt_ci(bias_ci, '°F')}   (+ = we forecast too COOL, − = too WARM)")

    # per-city bias — the grid-vs-sensor discovery, quantified
    by_city: dict[str, list[float]] = defaultdict(list)
    for (city, day), e in cd.items():
        by_city[city].append(e)
    print("\n    per-city bias (mean settled − forecast; flags |bias| ≥ 1.5°F):")
    flagged = []
    for city, errs in sorted(by_city.items(), key=lambda x: -abs(sum(x[1]) / len(x[1]))):
        mb = sum(errs) / len(errs)
        mae = sum(abs(e) for e in errs) / len(errs)
        flag = ""
        if abs(mb) >= 1.5:
            flag = "  ← WARM lean" if mb < 0 else "  ← COOL lean"
            flagged.append((city, mb))
        print(f"      {city:<5} n={len(errs):<3} bias={mb:+.1f}°F  MAE={mae:.1f}°F{flag}")

    # ── Q2  PROBABILITY EDGE vs MARKET ────────────────────────────────
    usable = [r for r in rows if r.get("prob_estimate") is not None
              and r.get("market_mid") is not None and r.get("yes_settled") is not None]
    mb_model: dict[str, list[float]] = defaultdict(list)
    mb_mkt: dict[str, list[float]] = defaultdict(list)
    diff_by_day: dict[str, list[float]] = defaultdict(list)
    for r in usable:
        y = 1.0 if r["yes_settled"] else 0.0
        bm = (float(r["prob_estimate"]) - y) ** 2
        bk = (float(r["market_mid"]) - y) ** 2
        d = r["settlement_date"]
        mb_model[d].append(bm)
        mb_mkt[d].append(bk)
        diff_by_day[d].append(bk - bm)          # >0 means MODEL beats market (lower Brier)
    bm_ci = _block_ci(mb_model)
    bk_ci = _block_ci(mb_mkt)
    diff_ci = _block_ci(diff_by_day)
    print(f"\n{'─'*70}\nQ2  PROBABILITY EDGE — Brier(model) vs Brier(market), {len(usable)} contracts")
    print(f"    model  Brier = {_fmt_ci(bm_ci)}")
    print(f"    market Brier = {_fmt_ci(bk_ci)}")
    print(f"    market − model (>0 = we beat the price) = {_fmt_ci(diff_ci)}")
    if diff_ci:
        p, lo, hi, k = diff_ci
        if k < _MIN_BLOCKS:
            print(f"    → UNPROVEN: only {k} independent days (< {_MIN_BLOCKS}).")
        elif lo > 0:
            print(f"    → we beat the market at 95% across {k} days.")
        elif hi < 0:
            print(f"    → the MARKET beats us at 95% — our probabilities are worse than the price.")
        else:
            print(f"    → indistinguishable from the market (CI spans 0).")

    # ── Q3  DIAGNOSIS: is the miscalibration just too-tight sigma? ─────
    # Leave-one-DAY-out per-city residual sigma → re-score Brier with a Gaussian
    # centered on the same forecast. In-sample-ish diagnostic; forward-validate before trusting.
    resid_city_day: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (city, day), e in cd.items():
        resid_city_day[city][day].append(e)
    diagB_by_day: dict[str, list[float]] = defaultdict(list)
    base_by_day: dict[str, list[float]] = defaultdict(list)
    import statistics
    for r in usable:
        city, day = r["city"], r["settlement_date"]
        pool = [e for dd, es in resid_city_day.get(city, {}).items() if dd != day for e in es]
        if len(pool) < 5:
            continue
        sigma = statistics.pstdev(pool) or 1e-6
        center = float(r["tmax_f_fcst"])
        p_new = nws_normal_prob(r.get("strike_type", ""), r.get("floor_strike"),
                                r.get("cap_strike"), center, sigma)
        if p_new is None:
            continue
        y = 1.0 if r["yes_settled"] else 0.0
        diagB_by_day[day].append((p_new - y) ** 2)
        base_by_day[day].append((float(r["prob_estimate"]) - y) ** 2)
    if diagB_by_day:
        diag_ci = _block_ci(diagB_by_day)
        base_ci = _block_ci(base_by_day)
        # realized sigma vs a rough sense of what the model used
        all_resid = [e for es in by_city.values() for e in es]
        print(f"\n{'─'*70}\nQ3  DIAGNOSIS — recalibrated spread (leave-one-day-out σ from residuals)")
        print(f"    realized residual σ (all cities) = {statistics.pstdev(all_resid):.1f}°F")
        print(f"    current-model Brier (same subset) = {_fmt_ci(base_ci)}")
        print(f"    widened-σ diagnostic Brier        = {_fmt_ci(diag_ci)}")
        print("    → If widened-σ Brier is materially lower, the forecast is fine and the")
        print("      bug is variance (overconfidence). This is an IN-SAMPLE diagnostic —")
        print("      present it as 'identified fix, to be forward-validated', not as achieved.")

    print(f"\n{'='*70}\n  Bottom line to frame from these numbers, not around them.\n{'='*70}\n")


if __name__ == "__main__":
    main()
