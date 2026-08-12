"""
Fit the probability-recalibration map (backlog P1.1 — fix YES over-prediction).

Reads graded outcomes (data/outcomes/signal_outcomes.jsonl), fits a monotone isotonic
map raw-P(YES) → realized-YES per `prob_source`, and saves the interpolation knots to
data/models/prob_calibration.json for calibration/recalibrate.py to apply at runtime.

Guardrails (CLAUDE.md):
  • No leakage: the map's benefit is proven on a TIME-SPLIT holdout (fit on the older
    portion, measure Brier + reliability on the newer). The production artifact is then
    refit on ALL data and applied forward (forward application is inherently OOS).
  • Protect the ensemble: a source is only fit if it has ≥ MIN_FIT_N graded rows; the
    ensemble path (small n, already best-calibrated) stays IDENTITY until it qualifies.

Usage:
    .venv/bin/python scripts/build_calibration_map.py [--min-n 150] [--test-frac 0.3]
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

from sklearn.isotonic import IsotonicRegression  # noqa: E402

OUTCOMES = ROOT / "data" / "outcomes" / "signal_outcomes.jsonl"
OUT_PATH = ROOT / "data" / "models" / "prob_calibration.json"


def _load_rows() -> list[dict]:
    rows = []
    for line in OUTCOMES.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # raw model output: prefer prob_raw (logged post-calibration era); fall back to
        # prob_estimate (pre-calibration rows, where prob_estimate IS the raw output).
        x = r.get("prob_raw", r.get("prob_estimate"))
        y = r.get("yes_settled")
        if x is None or y is None:
            continue
        rows.append({
            "x": float(x),
            "y": 1.0 if y else 0.0,
            "source": r.get("prob_source") or "unlabeled",
            "sdate": r.get("settlement_date", ""),
            "ticker": r.get("ticker", ""),
            "run_ts": r.get("run_ts", ""),
            "h2s": r.get("hours_to_settle"),
        })
    return rows


def _lead_population(rows: list[dict], source: str, lead: str) -> list[dict]:
    """
    The monitor-matched fit population for a lead-keyed map: rows of `source`
    whose horizon falls in the lead bucket, deduped to the LAST read per ticker
    (cycle-level re-logs of the same contract are one bet, not more evidence).
    MUST mirror scripts/skill_monitor.py — the map is fit on the same population
    the alarm judges.
    """
    from kalshi_weather.calibration.recalibrate import lead_bucket
    sub = [r for r in rows
           if r["source"] == source and lead_bucket(r["h2s"]) == lead]
    last: dict[str, dict] = {}
    for r in sorted(sub, key=lambda r: r["run_ts"]):
        last[r["ticker"]] = r
    return sorted(last.values(), key=lambda r: r["sdate"])


def _lodo_gate(pop: list[dict], n_bins: int = 8,
               min_n: int = 150, min_days: int = 7,
               min_day_win_frac: float = 0.6,
               min_rel_gain: float = 0.02) -> tuple[bool, dict]:
    """
    Leave-one-DAY-out proof that a fit on this population beats identity out of
    sample. Days, not rows — same-day outcomes are one weather system. The map
    ships only if the LODO weighted Brier improves by ≥ min_rel_gain RELATIVE
    (isotonic-as-shrinkage shaves <1% on even perfectly-calibrated noise — an
    epsilon map is indistinguishable from overfit; the real day-ahead effect is
    ~15%) AND a majority of held-out days individually improve.
    Returns (passed, evidence).
    """
    days = sorted({r["sdate"] for r in pop})
    ev: dict = {"n": len(pop), "days": len(days)}
    if len(pop) < min_n or len(days) < min_days:
        ev["why"] = f"insufficient sample (n={len(pop)}, days={len(days)})"
        return False, ev
    briers_id, briers_fit, wins = [], [], 0
    for hold in days:
        tr = [r for r in pop if r["sdate"] != hold]
        te = [r for r in pop if r["sdate"] == hold]
        iso = _fit_isotonic([r["x"] for r in tr], [r["y"] for r in tr], n_bins=n_bins)
        xt, yt = np.asarray([r["x"] for r in te]), np.asarray([r["y"] for r in te])
        b_id, b_fit = _brier(xt, yt), _brier(iso.predict(xt), yt)
        briers_id.append((b_id, len(te))); briers_fit.append((b_fit, len(te)))
        if b_fit < b_id - 1e-9:
            wins += 1
    tot = sum(n for _, n in briers_id)
    w_id = sum(b * n for b, n in briers_id) / tot
    w_fit = sum(b * n for b, n in briers_fit) / tot
    ev.update({"lodo_brier_identity": round(w_id, 4), "lodo_brier_fit": round(w_fit, 4),
               "days_improved": wins, "rel_gain": round(1.0 - w_fit / w_id, 4) if w_id else 0.0})
    passed = (w_fit < w_id * (1.0 - min_rel_gain)) and (wins / len(days) >= min_day_win_frac)
    if not passed:
        ev["why"] = "LODO does not beat identity by the required margin"
    return passed, ev


def _brier(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean((x - y) ** 2)) if len(x) else float("nan")


def _reliability(x, y, edges=(0, 0.2, 0.4, 0.6, 0.8, 1.01)) -> str:
    x, y = np.asarray(x), np.asarray(y)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum():
            out.append(f"    [{lo:.1f},{hi:.1f}) n={int(m.sum()):<4} pred≈{x[m].mean():.2f} real={y[m].mean():.2f}")
    return "\n".join(out)


def _fit_isotonic(x, y, n_bins: int = 12) -> IsotonicRegression:
    """
    Isotonic fit on QUANTILE-BINNED stats (bin mean-pred → bin realized-rate, weighted
    by count) rather than raw points. Binning guarantees each knot is backed by ~n/n_bins
    samples, so a single lucky win in the sparse high tail can't spike the map to 1.0
    (the raw-point fit did exactly that: x=0.86→y=1.0 while the 0.8–1.0 bin realizes 0.47).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    if len(edges) < 3:                       # too few distinct probs → fit raw
        iso.fit(x, y)
        return iso
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    bx, by, bw = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            bx.append(x[m].mean()); by.append(y[m].mean()); bw.append(int(m.sum()))
    iso.fit(np.asarray(bx), np.asarray(by), sample_weight=np.asarray(bw))
    return iso


def _knots(iso: IsotonicRegression) -> dict:
    xs = np.asarray(iso.X_thresholds_, dtype=float)
    ys = np.asarray(iso.y_thresholds_, dtype=float)
    # ensure endpoints span [0,1] so np.interp never extrapolates oddly
    return {"x": xs.round(5).tolist(), "y": ys.round(5).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=150, help="min graded rows to fit a source")
    ap.add_argument("--test-frac", type=float, default=0.3, help="holdout fraction (by date) for OOS proof")
    args = ap.parse_args()

    rows = _load_rows()
    if not rows:
        print("No graded outcomes with probabilities — run scripts/daily_review.py first.")
        return
    rows.sort(key=lambda r: r["sdate"])
    print(f"Loaded {len(rows)} graded rows.\n")

    # sources to consider going forward + pooled
    from collections import Counter
    src_n = Counter(r["source"] for r in rows)
    print("Rows per source:", dict(src_n), "\n")

    # ── OOS proof: time split ───────────────────────────────────────────────────
    cut = int(len(rows) * (1 - args.test_frac))
    train, test = rows[:cut], rows[cut:]
    print(f"OOS time-split: train n={len(train)} (≤{train[-1]['sdate']}), "
          f"test n={len(test)} (≥{test[0]['sdate']})")

    def _slice(data, source=None):
        d = [r for r in data if (source is None or r["source"] == source)]
        return [r["x"] for r in d], [r["y"] for r in d]

    # Pooled OOS: fit on all train, apply to all test.
    xtr, ytr = _slice(train)
    xte, yte = _slice(test)
    iso_pool = _fit_isotonic(xtr, ytr)
    raw_brier = _brier(xte, yte)
    cal_brier = _brier(iso_pool.predict(np.asarray(xte)), yte)
    print(f"\nPOOLED test Brier:  raw={raw_brier:.4f}  →  calibrated={cal_brier:.4f}  "
          f"({'better' if cal_brier < raw_brier else 'WORSE'})")
    print("  test reliability RAW:\n" + _reliability(xte, yte))
    print("  test reliability CALIBRATED:\n" + _reliability(iso_pool.predict(np.asarray(xte)), yte))

    # ── Production maps: refit on ALL data, per eligible source ─────────────────
    maps: dict = {"sources": {}, "meta": {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "n_total": len(rows),
        "min_fit_n": args.min_n,
        "oos_pooled_brier_raw": round(raw_brier, 4),
        "oos_pooled_brier_cal": round(cal_brier, 4),
        "notes": [],
    }}

    for source, n in src_n.items():
        if source == "ensemble":
            maps["meta"]["notes"].append(
                f"ensemble ({n}) base key left IDENTITY — protected; lead-keyed fit below")
            continue
        if n < args.min_n:
            maps["meta"]["notes"].append(f"{source} ({n}) < min_n → identity")
            continue
        xs, ys = _slice(rows, source)
        iso = _fit_isotonic(xs, ys)
        maps["sources"][source] = _knots(iso)
        b_raw, b_cal = _brier(xs, ys), _brier(iso.predict(np.asarray(xs)), ys)
        print(f"\nFit '{source}' (n={n}): in-sample Brier {b_raw:.4f} → {b_cal:.4f}")

    # ── Lead-keyed ensemble day-ahead map (evidence-gated) ──────────────────────
    # The day-ahead board systematically over-predicts (skill_monitor bias +0.053
    # CI excluding 0) while same-day is fine (−0.006) — so the correction is keyed
    # "ensemble@day-ahead" and ships ONLY when a leave-one-day-out proof shows the
    # fit beats identity on held-out days. Same-day stays identity deliberately.
    N_BINS_LEAD = 8   # ~27 rows/knot at n≈216 — coarser than the pooled fit on purpose
    pop = _lead_population(rows, "ensemble", "day-ahead")
    passed, ev = _lodo_gate(pop, n_bins=N_BINS_LEAD)
    if passed:
        iso = _fit_isotonic([r["x"] for r in pop], [r["y"] for r in pop],
                            n_bins=N_BINS_LEAD)
        maps["sources"]["ensemble@day-ahead"] = _knots(iso)
        maps["meta"]["ensemble_day_ahead_evidence"] = ev
        print(f"\nFit 'ensemble@day-ahead' (n={ev['n']}, days={ev['days']}): "
              f"LODO Brier {ev['lodo_brier_identity']:.4f} → {ev['lodo_brier_fit']:.4f}, "
              f"{ev['days_improved']}/{ev['days']} held-out days improved → SHIPPED")
    else:
        maps["meta"]["notes"].append(
            f"ensemble@day-ahead NOT emitted — {ev.get('why', 'gate failed')} "
            f"(evidence: {ev})")
        print(f"\n'ensemble@day-ahead' gate FAILED → identity kept ({ev})")

    # Fold historical 'unlabeled' into a pooled default under 'rule' if rule wasn't fit,
    # so the forward rule path always has a correction available.
    if "rule" not in maps["sources"]:
        maps["sources"]["rule"] = _knots(iso_pool)
        maps["meta"]["notes"].append("rule had <min_n alone → using pooled map")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(maps, indent=2))
    print(f"\nSaved → {OUT_PATH}")
    print("Sources calibrated:", list(maps["sources"].keys()))
    for note in maps["meta"]["notes"]:
        print("  ·", note)


if __name__ == "__main__":
    main()
