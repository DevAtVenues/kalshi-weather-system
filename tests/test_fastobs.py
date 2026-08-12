"""fastobs band math + fast_obs_watch stage logic (pure functions, no network)."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kalshi_weather.ingest.fastobs import (  # noqa: E402
    band_from_rows, c_to_f, is_precise_c,
)
from fast_obs_watch import SLACK_F, stage_for  # noqa: E402

D = date(2026, 7, 21)


def _row(hhmm: str, temp_c, raw=""):
    ts = pd.Timestamp(f"2026-07-21T{hhmm}:00Z")
    return {"ts": ts, "temp_c": temp_c, "raw": raw}


class TestPrecision:
    def test_tenths_value_is_precise(self):
        assert is_precise_c(36.7)

    def test_integer_value_is_band(self):
        assert not is_precise_c(37.0)
        assert not is_precise_c(37)

    def test_conversion(self):
        assert c_to_f(36.7) == pytest.approx(98.06)


class TestBand:
    def test_integer_c_is_a_band_not_a_point(self):
        # The 2026-07-21 SAT illusion: "37 C" looked like 98.6F but only
        # guarantees 97.7F. Band = [36.5, 37.5] C = [97.7, 99.5] F.
        b = band_from_rows([_row("20:45", 37.0)], -6, D)
        assert b["high_min_f"] == pytest.approx(97.7)
        assert b["high_max_f"] == pytest.approx(99.5)

    def test_tenths_dominate_the_floor(self):
        # hourly T-group 36.7 (98.06F exact) + 5-min integer 37 C
        b = band_from_rows([_row("20:51", 36.7, raw="KSAT ..."),
                            _row("21:00", 37.0)], -6, D)
        assert b["high_min_f"] == pytest.approx(98.06)   # tenths beat band floor
        assert b["high_max_f"] == pytest.approx(99.5)    # band ceiling stands
        assert b["latest_precise_f"] == pytest.approx(98.06)

    def test_lst_day_filter(self):
        # 04:00Z on Jul 22 UTC is still Jul 21 in LST-6 -> counted;
        # 07:00Z (01:00 LST Jul 22) is not.
        rows = [{"ts": pd.Timestamp("2026-07-22T04:00:00Z"), "temp_c": 30.0,
                 "raw": ""},
                {"ts": pd.Timestamp("2026-07-22T07:00:00Z"), "temp_c": 35.0,
                 "raw": ""}]
        b = band_from_rows(rows, -6, D)
        assert b["n_obs"] == 1
        assert b["high_max_f"] == pytest.approx(c_to_f(30.5))

    def test_no_rows_returns_none(self):
        assert band_from_rows([], -6, D) is None
        assert band_from_rows([_row("20:00", None)], -6, D) is None

    def test_flat_minutes(self):
        rows = [_row("19:51", 36.7, raw="x"), _row("20:51", 36.7, raw="x"),
                _row("21:51", 36.7, raw="x")]
        b = band_from_rows(rows, -6, D)
        assert b["flat_minutes"] == 120
        rows.append(_row("18:51", 35.0, raw="x"))       # older, different temp
        assert band_from_rows(rows, -6, D)["flat_minutes"] == 120

    def test_flat_breaks_on_change(self):
        rows = [_row("21:51", 36.7, raw="x"), _row("20:51", 36.2, raw="x")]
        assert band_from_rows(rows, -6, D)["flat_minutes"] == 0


def _entry(stt, direction, floor=None, cap=None):
    return {"ticker": "T", "strike_type": stt, "direction": direction,
            "floor": floor, "cap": cap}


def _band(hi_min, hi_max):
    return {"high_min_f": hi_min, "high_max_f": hi_max,
            "latest_precise_f": None, "flat_minutes": None}


class TestStages:
    # Canonical rules: B98.5 = {98, 99} INCLUSIVE — the settle zone in true-max
    # space is [97.5, 99.5); NO locks only past 99.5 (+0.5 slack -> 100.0).
    def test_between_no_quiet_warn_threat_win(self):
        e = _entry("between", "BUY_NO", floor=98, cap=99)
        assert stage_for(e, _band(95.0, 96.8)) is None
        assert stage_for(e, _band(96.8, 99.5)) == "warn"    # coarse feed in reach
        assert stage_for(e, _band(98.06, 99.5)) == "threat"  # tenths confirm
        assert stage_for(e, _band(100.0, 101.3)) == "win"    # past inclusive cap+slack

    def test_inside_inclusive_bracket_is_threat_not_win(self):
        # The 2026-07-22 KHOU lesson: 99.5 confirmed is still INSIDE {98,99}'s
        # reach + slack; only 100.0 locks the NO.
        e = _entry("between", "BUY_NO", floor=98, cap=99)
        assert stage_for(e, _band(99.0, 100.4)) == "threat"
        assert stage_for(e, _band(99.5, 100.4)) == "threat"
        assert stage_for(e, _band(100.0, 100.4)) == "win"

    def test_between_yes_dies_on_overshoot(self):
        e = _entry("between", "BUY_YES", floor=98, cap=99)
        assert stage_for(e, _band(99.5, 100.4)) is None      # 99 still possible
        assert stage_for(e, _band(100.0, 100.4)) == "dead"

    def test_greater_yes_near_and_win(self):
        e = _entry("greater", "BUY_YES", floor=96)   # strict: YES needs >= 97
        assert stage_for(e, _band(93.0, 94.1)) is None
        assert stage_for(e, _band(94.0, 95.9)) == "near"
        assert stage_for(e, _band(97.0, 97.7)) == "win"
        assert stage_for(e, _band(96.5, 97.7)) == "near"     # == floor+slack: not yet

    def test_greater_no_dies_when_floor_cleared(self):
        e = _entry("greater", "BUY_NO", floor=96)
        assert stage_for(e, _band(97.0, 97.7)) == "dead"

    def test_less_no_wins_past_cap(self):
        e = _entry("less", "BUY_NO", cap=94)
        assert stage_for(e, _band(94.5, 95.0)) == "win"
        assert stage_for(e, _band(93.9, 95.0)) is None
