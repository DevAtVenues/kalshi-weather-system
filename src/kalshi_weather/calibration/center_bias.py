"""
Per-city ensemble center correction (backlog P1.0) — shrinkage toward a neutral prior.

The 71-member ensemble (GEFS + ECMWF ENS + ICON) has city-specific systematic biases
versus the NWS CLI settlement temperature. We correct for this bias using:
  • MEASURED — the live ensemble-center error (actual − ens_p50), the RIGHT quantity,
    logged since 2026-07-07.
  • NEUTRAL PRIOR — 0.0°F (no systematic bias assumed until we measure one).

We James-Stein-shrink the measured shift toward 0 (not toward the historical GFS-vs-CLI
bias). Early empirical data (July 2026, 2 days per city) showed the GFS single-run prior
disagreed in sign with the ensemble for KNYC (+0.85°F measured vs -0.72°F GFS prior) and
KPHL (-1.15°F vs +0.92°F). The GFS and ensemble represent the grid differently, use
different model physics, and have independent grid-to-sensor artifacts — using GFS as an
ensemble prior contaminated the correction and pointed it in the wrong direction.

Neutral prior means: with n=0 days, apply 0.0°F correction. With n=2 days and PRIOR_DAYS=5,
apply 2/7 weight to the measured, 5/7 weight to 0 → mild correction in the right direction.

IMPORTANT: _measured_shift uses ONLY the real ens_center_err column (actual − ens_p50),
never the GFS forecast_error_f proxy. The proxy conflates the deterministic single-run
forecast with the ensemble median — these can diverge significantly (the single GFS run
is often a tail member), especially during anomalous events like heat waves.

Wired to the live engine 2026-07-09 (build_ensemble_context in live/runner.py).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from kalshi_weather.calibration.bias import load_bias_table

_OUTCOMES = Path(__file__).parents[3] / "data" / "outcomes" / "signal_outcomes.jsonl"
PRIOR_DAYS = 5.0    # neutral-prior shrinkage weight; 5 days of inertia before measured takes over


def _measured_shift(station: str, month: int) -> tuple[float | None, int]:
    """Mean per-day ensemble-center shift (actual − ens_p50) for this station+month, and
    the number of distinct days with real ens_p50 data.

    Only uses genuine ens_center_err (actual − ens_p50 logged by the live engine) — NOT
    the GFS deterministic forecast_error_f proxy. The proxy was previously used to fill
    missing values but conflates two different quantities: the deterministic GFS single
    run (a potential tail member) vs. the 71-member ensemble median. During anomalous
    events (e.g. heat waves) these diverge substantially and the proxy contaminates the
    prior, potentially reversing the direction of the correction.

    Deduplicates to the LATEST run_ts per ticker before averaging — the outcomes file
    accumulates many snapshots of each contract as the engine re-prices throughout the
    day. Each new run may produce a different ens_p50 (new model run available), so
    using all snapshots would weight early-morning stale estimates and under-measure
    the true daily bias. The latest snapshot is what the engine committed to.
    """
    if not _OUTCOMES.exists():
        return None, 0
    # Step 1: deduplicate to latest run_ts per ticker
    latest: dict[str, dict] = {}
    for line in _OUTCOMES.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("prob_source") != "ensemble" or r.get("station") != station:
            continue
        sd = r.get("settlement_date", "")
        if not sd or int(sd[5:7]) != month:
            continue
        if r.get("ens_center_err") is None:
            continue
        ticker = r.get("ticker", "")
        run_ts = str(r.get("run_ts", ""))
        if ticker not in latest or run_ts > str(latest[ticker].get("run_ts", "")):
            latest[ticker] = r
    if not latest:
        return None, 0
    # Step 2: group by settlement_date and compute per-day mean
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in latest.values():
        by_day[r["settlement_date"]].append(float(r["ens_center_err"]))
    day_means = [sum(v) / len(v) for v in by_day.values()]
    return sum(day_means) / len(day_means), len(day_means)


def center_shift(station: str, month: int, bias_table: dict | None = None) -> float:
    """°F to ADD to the ensemble forecast center for this station+month.

    James-Stein shrinks the measured ens-center shift toward a NEUTRAL PRIOR of 0.0°F.
    With 0 measured days: apply 0.0°F (no correction assumed). As forward data
    accumulates the correction strengthens. Returns 0.0 when no measured data exists.

    Note: bias_table is accepted but not used for the prior — the GFS historical bias
    is not a reliable prior for the ensemble center (different model, different grid
    physics, independent grid-to-sensor artifacts). The neutral prior prevents
    wrong-direction corrections when GFS and ensemble biases happen to disagree.
    """
    # neutral prior — ignore bias_table for ensemble center correction
    measured, n = _measured_shift(station, month)
    if measured is None or n == 0:
        return 0.0
    shrunk = (n * measured) / (n + PRIOR_DAYS)   # shrinks toward 0.0 (neutral prior)
    return round(shrunk, 2)
