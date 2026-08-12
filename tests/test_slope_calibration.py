"""Targeted-slope calibration (SFO/SEA) — center + spread behavior.

Verifies the OOS-validated linear correction is applied ONLY to SLOPE_STATIONS,
that interior cities keep the flat monthly offset, and that the sigma consumers
read (std_f) is swapped to the linear model's residual std for slope stations.
"""
import pandas as pd
import pytest

from kalshi_weather.calibration.bias import (
    SLOPE_STATIONS, apply_correction, load_bias_table,
)

bt = load_bias_table()
pytestmark = pytest.mark.skipif(
    not bt or "KSFO" not in bt,
    reason="bias.parquet with slope columns not built (run scripts/build_slope_calibration.py)",
)


def test_slope_stations_configured():
    assert SLOPE_STATIONS == {"KSFO", "KSEA"}


def test_slope_station_uses_linear_form():
    entry = bt["KSFO"][7]
    assert {"slope_a", "slope_b", "slope_std_f"} <= entry.keys()
    x = 78.0
    got = apply_correction(x, "KSFO", 7, bt)
    expected = round(entry["slope_a"] + entry["slope_b"] * x, 1)
    assert got == expected
    # linear form must actually differ from the flat offset (SFO July b≈0.58)
    assert got != round(x - entry["bias_f"], 1)


def test_slope_shrinks_warm_forecasts_toward_mean():
    # b < 1 → a hot forecast is pulled DOWN more than a mild one (marine layer cap)
    e = bt["KSFO"][7]
    assert e["slope_b"] < 0.95
    hot = apply_correction(90.0, "KSFO", 7, bt)
    mild = apply_correction(70.0, "KSFO", 7, bt)
    # corrected spread is compressed vs the 20°F raw spread
    assert (hot - mild) < 20.0


def test_slope_station_sigma_is_linear_residual_std():
    e = bt["KSFO"][7]
    # load_bias_table swaps std_f → the linear model's residual std for slope stations
    assert e["std_f"] == e["slope_std_f"]


def test_interior_station_keeps_flat_offset():
    assert "KDFW" not in SLOPE_STATIONS
    e = bt["KDFW"][7]
    x = 95.0
    assert apply_correction(x, "KDFW", 7, bt) == round(x - e["bias_f"], 1)
    # interior std_f is the raw error std, NOT swapped
    if "slope_std_f" in e:
        assert e["std_f"] != e["slope_std_f"] or e["slope_b"] == pytest.approx(1.0, abs=0.02)


def test_unknown_station_returns_none():
    assert apply_correction(80.0, "KXXX", 7, bt) is None
