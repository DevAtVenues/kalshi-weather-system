"""Replay harness + provenance stamps: the model-behavior regression gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import replay_check as rc
from kalshi_weather.provenance import provenance

NOW_TS = "2099-01-01T00:00:00+00:00"    # newer than any map mtime → comparable


def _row(**kw):
    base = {"run_ts": NOW_TS, "ticker": "T-TEST", "prob_source": "ensemble",
            "prob_raw": 0.30, "prob_estimate": 0.30, "market_mid": 0.10,
            "edge_raw": 0.20, "direction": "BUY_YES"}
    base.update(kw)
    return base


def test_consistent_rows_pass():
    res = rc.check_rows([_row()], maps={}, current_hash=None, map_mtime=0.0)
    assert res["calib_checked"] == 1
    assert not (res["calib_drift"] or res["edge_drift"] or res["dir_drift"])


def test_calibration_drift_detected():
    # identity map but the logged estimate disagrees with the logged raw
    res = rc.check_rows([_row(prob_estimate=0.55, edge_raw=0.45)],
                        maps={}, current_hash=None, map_mtime=0.0)
    assert len(res["calib_drift"]) == 1


def test_direction_drift_detected():
    res = rc.check_rows([_row(direction="BUY_NO")],
                        maps={}, current_hash=None, map_mtime=0.0)
    assert len(res["dir_drift"]) == 1


def test_edge_drift_detected():
    res = rc.check_rows([_row(edge_raw=0.35)],
                        maps={}, current_hash=None, map_mtime=0.0)
    assert len(res["edge_drift"]) == 1


def test_older_artifact_rows_skipped_not_drifted():
    row = _row(calib_hash="oldhash1234", prob_estimate=0.55)  # would drift if compared
    res = rc.check_rows([row], maps={}, current_hash="newhash5678", map_mtime=0.0)
    assert res["calib_skipped_era"] == 1 and not res["calib_drift"]
    # Layer B still applies across eras (pure arithmetic):
    assert len(res["edge_drift"]) == 1   # 0.20 logged vs |0.55-0.10|


def test_map_applied_path_replays():
    maps = {"sources": {"rule": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}}
    # apply: interp(0.30)=0.15 → clip(0.01,0.97)=0.15
    ok = _row(prob_source="rule", prob_estimate=0.15, edge_raw=0.05)
    res = rc.check_rows([ok], maps=maps, current_hash=None, map_mtime=0.0)
    assert not res["calib_drift"]
    bad = _row(prob_source="rule", prob_estimate=0.30, edge_raw=0.20)  # calib not applied
    res = rc.check_rows([bad], maps=maps, current_hash=None, map_mtime=0.0)
    assert len(res["calib_drift"]) == 1


def test_provenance_stamp_shape():
    p = provenance()
    assert p["code_sha"] and p["code_sha"] != "unknown"
    assert p["calib_hash"] and len(p["calib_hash"]) == 10
    assert p["curve_hash"] and len(p["curve_hash"]) == 10
