"""Per-city ensemble center correction — shrinkage of prior + forward data."""
import json
import tempfile
from pathlib import Path

from kalshi_weather.calibration import center_bias


def test_no_measured_data_returns_zero(monkeypatch):
    # With no measured ens-center error, return 0.0 (neutral prior — GFS prior not used
    # because GFS single-run and ensemble biases are independent and can disagree in sign).
    monkeypatch.setattr(center_bias, "_measured_shift", lambda s, m: (None, 0))
    bt = {"KNYC": {7: {"bias_f": 0.72}}}
    assert center_bias.center_shift("KNYC", 7, bt) == 0.0


def test_shrinks_noisy_measurement_toward_zero(monkeypatch):
    # A wild measured shift (-3.0) on few days is pulled toward 0 (neutral prior).
    monkeypatch.setattr(center_bias, "_measured_shift", lambda s, m: (-3.0, 2))
    bt = {"KNYC": {7: {"bias_f": 0.72}}}
    shift = center_bias.center_shift("KNYC", 7, bt)
    # (2*-3.0)/(2+5) = -0.857 — shrunk heavily toward 0 with n=2, PRIOR_DAYS=5
    assert -1.1 < shift < -0.5


def test_measured_dominates_once_data_is_plentiful(monkeypatch):
    monkeypatch.setattr(center_bias, "_measured_shift", lambda s, m: (-3.0, 90))
    bt = {"KNYC": {7: {"bias_f": 0.72}}}
    shift = center_bias.center_shift("KNYC", 7, bt)
    assert shift < -2.7    # 90 days swamps the 5-day neutral prior → ≈ -2.84


def test_missing_station_returns_zero(monkeypatch):
    monkeypatch.setattr(center_bias, "_measured_shift", lambda s, m: (None, 0))
    assert center_bias.center_shift("KXXX", 7, {}) == 0.0


def test_bias_table_ignored_for_ensemble_prior(monkeypatch):
    # The GFS bias_table is NOT used as a prior — center_shift ignores it.
    # Two calls with same measured but different bias tables must return the same result.
    monkeypatch.setattr(center_bias, "_measured_shift", lambda s, m: (1.0, 3))
    bt_hot  = {"KNYC": {7: {"bias_f":  2.0}}}   # GFS ran very hot
    bt_cold = {"KNYC": {7: {"bias_f": -2.0}}}   # GFS ran very cold
    assert center_bias.center_shift("KNYC", 7, bt_hot) == center_bias.center_shift("KNYC", 7, bt_cold)


def test_measured_shift_deduplicates_to_latest_run_ts(monkeypatch, tmp_path):
    """Multiple snapshots of the same ticker on the same day must produce the same
    result as a single snapshot — only the latest run_ts per ticker is used.
    Without dedup, stale early-morning ens_p50 values pollute the day-mean."""
    # Ticker ABC-26JUL07 scored 3 times with different ens_center_err due to
    # updated ensemble runs. Only the last snapshot (run_ts "C", err=2.0) should count.
    rows = [
        {"ticker": "ABC-26JUL07", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-07", "run_ts": "A", "ens_center_err": -1.0},
        {"ticker": "ABC-26JUL07", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-07", "run_ts": "B", "ens_center_err":  0.5},
        {"ticker": "ABC-26JUL07", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-07", "run_ts": "C", "ens_center_err":  2.0},
        # A second ticker on the same day (also latest = 1.0)
        {"ticker": "DEF-26JUL07", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-07", "run_ts": "C", "ens_center_err":  1.0},
    ]
    out = tmp_path / "outcomes.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(center_bias, "_OUTCOMES", out)

    measured, n = center_bias._measured_shift("KNYC", 7)
    # After dedup: ABC→2.0, DEF→1.0 → day mean = 1.5 (not a contaminated average)
    assert n == 1, f"expected 1 distinct day, got {n}"
    assert measured is not None and abs(measured - 1.5) < 0.01, f"expected 1.5, got {measured}"


def test_measured_shift_counts_days_not_tickers(monkeypatch, tmp_path):
    """n returned by _measured_shift is DAYS, not tickers. Two tickers on the same day
    still count as 1 day for the James-Stein shrinkage denominator."""
    rows = [
        {"ticker": "T1-26JUL07", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-07", "run_ts": "Z", "ens_center_err": 1.0},
        {"ticker": "T2-26JUL07", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-07", "run_ts": "Z", "ens_center_err": 3.0},
        {"ticker": "T3-26JUL08", "prob_source": "ensemble", "station": "KNYC",
         "settlement_date": "2026-07-08", "run_ts": "Z", "ens_center_err": 0.0},
    ]
    out = tmp_path / "outcomes.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(center_bias, "_OUTCOMES", out)

    measured, n = center_bias._measured_shift("KNYC", 7)
    assert n == 2, f"expected 2 distinct days, got {n}"
    # day means: Jul7=(1.0+3.0)/2=2.0, Jul8=0.0 → grand mean=1.0
    assert measured is not None and abs(measured - 1.0) < 0.01, f"expected 1.0, got {measured}"
