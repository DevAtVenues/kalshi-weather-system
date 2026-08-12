"""
Lead-keyed probability recalibration (ensemble@day-ahead fix).

Pins: (a) the lead_bucket boundaries shared by runner / replay / builder /
skill_monitor, (b) the most-specific-first map lookup with identity fallbacks,
(c) the builder's LODO evidence gate — a lead map may only ship when it beats
identity on held-out DAYS (no manufactured edge), (d) replay compatibility.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]

from kalshi_weather.calibration.recalibrate import (  # noqa: E402
    apply_prob_calibration, lead_bucket)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_calibration_map", ROOT / "scripts" / "build_calibration_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── lead_bucket boundaries ─────────────────────────────────────────────────────

@pytest.mark.parametrize("h,expect", [
    (None, None), (float("nan"), None), ("junk", None),
    (-3.0, None), (0.0, None),           # window open / at boundary → intraday path
    (0.1, "same-day"), (5.0, "same-day"), (14.0, "same-day"),
    (14.001, "day-ahead"), (27.0, "day-ahead"), (40.0, "day-ahead"),
    (40.1, None), (72.0, None),          # beyond tradeable horizon
])
def test_lead_bucket(h, expect):
    assert lead_bucket(h) == expect


def test_lead_bucket_matches_skill_monitor_constants():
    """Fit population and alarm population must bucket identically."""
    spec = importlib.util.spec_from_file_location(
        "skill_monitor", ROOT / "scripts" / "skill_monitor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, lo, hi in mod.LEAD_BUCKETS:
        eps = 1e-6
        assert lead_bucket(lo + eps) == name
        assert lead_bucket(hi) == name
        assert lead_bucket(hi + eps) != name


# ── map lookup precedence ──────────────────────────────────────────────────────

MAPS = {"sources": {
    "ensemble@day-ahead": {"x": [0.0, 0.3, 1.0], "y": [0.0, 0.12, 0.5]},
    "rule": {"x": [0.0, 1.0], "y": [0.0, 0.5]},
}}


def test_lead_key_preferred():
    out = apply_prob_calibration(0.30, "ensemble", MAPS, lead="day-ahead")
    assert out == pytest.approx(0.12, abs=1e-9)


def test_falls_back_to_source_key_when_lead_unmapped():
    out = apply_prob_calibration(0.30, "rule", MAPS, lead="day-ahead")
    assert out == pytest.approx(0.15, abs=1e-9)   # rule map: 0.3 → 0.15


def test_identity_when_no_key_matches():
    # ensemble same-day: no "ensemble@same-day", no bare "ensemble" → IDENTITY,
    # un-clipped (an intraday 0.995 lock must survive).
    assert apply_prob_calibration(0.995, "ensemble", MAPS, lead="same-day") == 0.995
    assert apply_prob_calibration(0.995, "ensemble", MAPS, lead=None) == 0.995


def test_mapped_output_capped_below_certainty():
    maps = {"sources": {"ensemble@day-ahead": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}}
    out = apply_prob_calibration(0.999, "ensemble", maps, lead="day-ahead")
    assert out == pytest.approx(0.97)


def test_legacy_signature_unchanged():
    """Existing 3-arg callers (replay of pre-lead rows) keep working."""
    assert apply_prob_calibration(0.30, "rule", MAPS) == pytest.approx(0.15, abs=1e-9)


# ── builder: population + evidence gate ───────────────────────────────────────

def _mk_rows(n_days=10, per_day=25, bias=0.15, seed=7, source="ensemble"):
    """Graded outcomes where predicted P systematically exceeds realized rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        sdate = f"2026-06-{d + 1:02d}"
        for i in range(per_day):
            p_true = float(rng.uniform(0.02, 0.35))
            p_model = min(0.99, p_true + bias * p_true / 0.35)
            y = 1.0 if rng.uniform() < p_true else 0.0
            rows.append({"x": p_model, "y": y, "source": source, "sdate": sdate,
                         "ticker": f"T{d}-{i}", "run_ts": f"2026-06-{d + 1:02d}T12:00:00",
                         "h2s": 20.0})
    return rows


def test_lead_population_dedups_last_read_per_ticker():
    b = _load_builder()
    rows = [
        {"x": .2, "y": 0, "source": "ensemble", "sdate": "2026-06-01",
         "ticker": "A", "run_ts": "2026-06-01T06:00:00", "h2s": 30.0},
        {"x": .4, "y": 0, "source": "ensemble", "sdate": "2026-06-01",
         "ticker": "A", "run_ts": "2026-06-01T12:00:00", "h2s": 20.0},   # later read wins
        {"x": .3, "y": 1, "source": "ensemble", "sdate": "2026-06-01",
         "ticker": "B", "run_ts": "2026-06-01T12:00:00", "h2s": 5.0},    # same-day → excluded
        {"x": .3, "y": 1, "source": "rule", "sdate": "2026-06-01",
         "ticker": "C", "run_ts": "2026-06-01T12:00:00", "h2s": 20.0},   # wrong source
    ]
    pop = b._lead_population(rows, "ensemble", "day-ahead")
    assert [r["ticker"] for r in pop] == ["A"]
    assert pop[0]["x"] == .4


def test_lodo_gate_passes_on_systematic_bias():
    b = _load_builder()
    passed, ev = b._lodo_gate(_mk_rows(bias=0.20))
    assert passed, ev
    assert ev["lodo_brier_fit"] < ev["lodo_brier_identity"]


def test_lodo_gate_rejects_calibrated_data():
    b = _load_builder()
    passed, ev = b._lodo_gate(_mk_rows(bias=0.0))
    assert not passed, ev


def test_lodo_gate_rejects_insufficient_sample():
    b = _load_builder()
    passed, ev = b._lodo_gate(_mk_rows(n_days=4, per_day=10))
    assert not passed
    assert "insufficient" in ev["why"]


# ── replay compatibility ───────────────────────────────────────────────────────

def test_replay_derives_lead_for_legacy_rows():
    """replay_check falls back to lead_bucket(hours_to_settle) for pre-calib_lead rows."""
    assert lead_bucket(15.0) == "day-ahead"
    legacy = apply_prob_calibration(0.30, "ensemble", MAPS, lead=lead_bucket(15.0))
    assert legacy == pytest.approx(0.12, abs=1e-9)
