"""Guard against silent ensemble-feed degradation (2026-07-30 incident).

ecmwf_ifs04 was deprecated upstream and returned ZERO members while GFS+ICON
kept the total member count plausible, so every consumer priced off a
distribution missing its best model. fetch_ensemble_maxes must refuse a
partial ensemble: any model family below its member floor -> empty array.
"""
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from kalshi_weather.calibration import ensemble_dist as ed


def _fake_response(families: dict[str, int]) -> dict:
    """Build an API payload with `count` member series per family key."""
    hourly: dict = {"time": ["2026-07-30T00:00", "2026-07-30T01:00"]}
    for fam, count in families.items():
        for i in range(count):
            hourly[f"temperature_2m_member{i:02d}_{fam}_seamless"] = [90.0, 91.0 + i % 3]
    return {"hourly": hourly}


def _fetch_with(payload: dict, cache_dir=None) -> np.ndarray:
    """Run a mocked fetch against an isolated cache dir (fetch also READS the
    cache now, so leakage between tests would mask degraded-feed failures)."""
    resp = mock.Mock()
    resp.json.return_value = payload
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(ed, "_MEMBER_CACHE", Path(cache_dir or tmp)):
            with mock.patch.object(ed.requests, "get", return_value=resp):
                return ed.fetch_ensemble_maxes(25.8, -80.3, "2026-07-30", "America/New_York")


def test_full_ensemble_passes():
    out = _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}))
    assert out.size == 31 + 51 + 40


def test_missing_ecmwf_fails_whole_fetch(capsys):
    out = _fetch_with(_fake_response({"ncep_gefs": 31, "icon_eps": 40}))
    assert out.size == 0
    assert "CRITICAL" in capsys.readouterr().err


def test_thin_family_fails_whole_fetch():
    # A family present but far below its floor is the same failure.
    out = _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 5, "icon_eps": 40}))
    assert out.size == 0


def test_requested_models_use_current_api_names():
    # ifs04 is deprecated; requesting it yields zero ECMWF members silently.
    assert "ifs04" not in ed._MODELS
    assert "ecmwf_ifs025" in ed._MODELS


def test_fetch_writes_family_cache_and_reader_roundtrips(tmp_path):
    with mock.patch.object(ed, "_MEMBER_CACHE", tmp_path):
        out = _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}), tmp_path)
        assert out.size == 122
        c = ed.cached_ensemble_maxes(25.8, -80.3, "2026-07-30")
        assert c is not None and c["age_min"] < 1
        assert {f for f in c["families"]} == {"gefs", "ecmwf", "icon"}
        assert c["families"]["ecmwf"].size == 51
        # age gate
        assert ed.cached_ensemble_maxes(25.8, -80.3, "2026-07-30", max_age_min=0.0) is None


def test_fresh_cache_short_circuits_http(tmp_path):
    # Second fetch inside the TTL must serve from cache: 30-min engine cadence
    # re-downloading identical model cycles is what exhausted the daily quota.
    with mock.patch.object(ed, "_MEMBER_CACHE", tmp_path):
        out = _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}), tmp_path)
        assert out.size == 122
        with mock.patch.object(ed.requests, "get",
                               side_effect=AssertionError("HTTP hit inside TTL")):
            again = ed.fetch_ensemble_maxes(25.8, -80.3, "2026-07-30", "America/New_York")
        assert again.size == 122


def test_stale_cache_refetches(tmp_path):
    with mock.patch.object(ed, "_MEMBER_CACHE", tmp_path):
        _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}), tmp_path)
        # Age the cache past every TTL, then confirm the API is consulted again.
        p = ed._cache_path(25.8, -80.3, "2026-07-30")
        j = json.loads(p.read_text())
        j["fetch_ts"] = "2026-07-29T00:00:00+00:00"
        p.write_text(json.dumps(j))
        out = _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}), tmp_path)
        assert out.size == 122
        assert json.loads(p.read_text())["fetch_ts"] != "2026-07-29T00:00:00+00:00"


def test_cache_below_floor_is_not_served(tmp_path):
    # A cache written under older, laxer floors must not satisfy today's guard.
    with mock.patch.object(ed, "_MEMBER_CACHE", tmp_path):
        _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}), tmp_path)
        p = ed._cache_path(25.8, -80.3, "2026-07-30")
        j = json.loads(p.read_text())
        j["families"]["ecmwf"] = j["families"]["ecmwf"][:5]
        p.write_text(json.dumps(j))
        out = _fetch_with(_fake_response({"ncep_gefs": 31, "ecmwf_ifs025_x": 51, "icon_eps": 40}), tmp_path)
        assert out.size == 122  # served by a fresh fetch, not the crippled cache


def test_failed_fetch_does_not_write_cache(tmp_path):
    with mock.patch.object(ed, "_MEMBER_CACHE", tmp_path):
        out = _fetch_with(_fake_response({"ncep_gefs": 31}), tmp_path)   # ecmwf+icon missing
        assert out.size == 0
        assert ed.cached_ensemble_maxes(25.8, -80.3, "2026-07-30") is None
