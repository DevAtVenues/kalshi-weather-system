"""Tests for scripts/ensemble_mae.py — the corrected ensemble's real accuracy.

Two things these guard, beyond arithmetic:
  * the filters that keep the number honest (post-window rows excluded, one
    observation per city-day, ensemble-priced only);
  * that the bootstrap does not repeat backtest.py's degeneracy (audit F31).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("ensemble_mae", ROOT / "scripts" / "ensemble_mae.py")
em = importlib.util.module_from_spec(spec)
sys.modules["ensemble_mae"] = em
spec.loader.exec_module(em)


def _row(city="NYC", day="2026-07-15", *, actual=90.0, p50=88.0, gfs=86.0,
         lead=20.0, run_ts="2026-07-14T12:00:00+00:00", source="ensemble",
         station="KNYC"):
    return {"city": city, "station": station, "settlement_date": day,
            "actual_temp": actual, "ens_p50": p50, "tmax_f_fcst": gfs,
            "hours_to_settle": lead, "run_ts": run_ts, "prob_source": source}


# ── arithmetic ──────────────────────────────────────────────────────────────

def test_error_is_actual_minus_forecast_for_both_centres():
    e = em.errors([_row(actual=90.0, p50=88.0, gfs=86.0)])[0]
    assert e["ens_err"] == pytest.approx(2.0)     # we forecast 2F COOL
    assert e["gfs_err"] == pytest.approx(4.0)


def test_mae_and_bias_are_distinct():
    vals = [2.0, -2.0, 2.0, -2.0]
    assert em.mae(vals) == pytest.approx(2.0)
    assert em.bias(vals) == pytest.approx(0.0)


# ── the filters that keep it honest ─────────────────────────────────────────

def test_post_window_rows_are_excluded_by_default():
    """On the settlement day ens_p50 is conditioned on the observed running high
    (runner.py:515), so including those rows flatters the MAE."""
    rows = [_row(lead=20.0), _row(day="2026-07-16", lead=-3.0)]
    assert len(em.usable(rows, 0.0, None)) == 1


def test_min_lead_cut_is_respected():
    rows = [_row(lead=5.0), _row(day="2026-07-16", lead=20.0)]
    assert len(em.usable(rows, 14.0, None)) == 1


def test_non_ensemble_rows_are_excluded():
    rows = [_row(), _row(day="2026-07-16", source="rule")]
    assert len(em.usable(rows, 0.0, None)) == 1


def test_rows_missing_either_centre_are_excluded():
    rows = [_row(), _row(day="a", p50=None), _row(day="b", actual=None)]
    assert len(em.usable(rows, 0.0, None)) == 1


def test_nan_lead_is_excluded_rather_than_silently_passing():
    """NaN compares False against everything — the exact shape that turns a
    missing field into an accidental pass elsewhere in this repo (audit F8)."""
    assert em.usable([_row(lead=float("nan"))], 0.0, None) == []


def test_one_observation_per_city_day_keeping_the_last_decision():
    """run_live logs the same contract every 30 min; counting each snapshot
    inflates n by the number of cycles a pick survived (audit F9)."""
    rows = [_row(run_ts="2026-07-14T10:00:00+00:00", p50=80.0),
            _row(run_ts="2026-07-14T18:00:00+00:00", p50=89.0),
            _row(city="CHI", p50=85.0)]
    kept = em.per_city_day(rows)
    assert len(kept) == 2
    nyc = [r for r in kept if r["city"] == "NYC"][0]
    assert nyc["ens_p50"] == pytest.approx(89.0)      # the later snapshot


def test_station_filter():
    rows = [_row(station="KNYC"), _row(city="CHI", station="KMDW")]
    assert len(em.usable(rows, 0.0, "KNYC")) == 1


# ── the bootstrap ───────────────────────────────────────────────────────────

def _errs(n_days, per_day=2, seed=3):
    import random
    rng = random.Random(seed)
    return [{"day": f"d{d}", "ens_err": rng.gauss(0.5, 2.0), "gfs_err": rng.gauss(1.0, 2.5)}
            for d in range(n_days) for _ in range(per_day)]


@pytest.mark.parametrize("n_days", [5, 10, 15, 30, 52, 60, 90])
def test_bootstrap_interval_is_never_degenerate(n_days):
    """backtest.py:269 returns a ZERO-WIDTH interval whenever the distinct-day
    count divides 30 (audit F31). A day-clustered resample must not."""
    pt, lo, hi = em.day_bootstrap(_errs(n_days), "ens_err", em.mae, n_boot=400)
    assert lo < pt < hi, f"degenerate interval at {n_days} days"
    assert hi - lo > 1e-6


def test_bootstrap_interval_narrows_with_more_days():
    w = []
    for n in (10, 100):
        _, lo, hi = em.day_bootstrap(_errs(n), "ens_err", em.mae, n_boot=800)
        w.append(hi - lo)
    assert w[1] < w[0]


def test_bootstrap_resamples_days_not_rows():
    """All rows in a day must move together, or within-day weather correlation
    is ignored and the interval comes out far too tight."""
    errs = [{"day": "d0", "ens_err": 0.0}] * 50 + [{"day": "d1", "ens_err": 10.0}] * 50
    pt, lo, hi = em.day_bootstrap(errs, "ens_err", em.bias, n_boot=2000)
    # Resampling 2 days with replacement gives means of 0, 5 or 10 — a wide
    # interval. Row-level resampling would concentrate tightly around 5.
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(10.0, abs=1e-6)


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    a = em.day_bootstrap(_errs(20), "ens_err", em.mae, n_boot=300, seed=1)
    b = em.day_bootstrap(_errs(20), "ens_err", em.mae, n_boot=300, seed=1)
    assert a == b


def test_single_day_returns_a_point_without_a_fake_interval():
    pt, lo, hi = em.day_bootstrap(_errs(1), "ens_err", em.mae, n_boot=100)
    assert pt == pt                       # not NaN
    assert lo != lo and hi != hi          # NaN, not a fabricated zero-width CI


# ── loading ─────────────────────────────────────────────────────────────────

def test_missing_outcomes_file_explains_itself(tmp_path):
    with pytest.raises(SystemExit) as e:
        em.load(tmp_path / "nope.jsonl")
    assert "grade_signals" in str(e.value)


def test_malformed_lines_are_skipped(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps(_row()) + "\n{ not json\n\n" + json.dumps(_row(city="CHI")) + "\n")
    assert len(em.load(p)) == 2
