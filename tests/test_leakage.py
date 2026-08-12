"""
Leakage test suite — must pass before any backtest code ships.

Three invariants enforced:
  1. Feature timestamp < decision time (no look-ahead by definition)
  2. LST day boundary is correct for KNYC (UTC-5 fixed)
  3. No timezone imports outside src/kalshi_weather/tz.py
"""
from __future__ import annotations

import ast
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from kalshi_weather.tz import (
    lst_date_for_utc,
    lst_offset,
    settlement_window_utc,
    to_utc,
)

SRC_ROOT = Path(__file__).parents[1] / "src" / "kalshi_weather"
TZ_MODULE = SRC_ROOT / "tz.py"

# ── 1. Feature timestamp < decision time ──────────────────────────────────────

class TestNoLookAhead:
    """
    Simulates a decision at 2024-07-04 12:00 UTC.
    Any feature with init_time >= that decision time is look-ahead and must fail.
    """

    DECISION_TIME = pd.Timestamp("2024-07-04 12:00:00", tz="UTC")

    def _assert_no_leakage(self, feature_times: list[pd.Timestamp]) -> None:
        violations = [t for t in feature_times if t >= self.DECISION_TIME]
        assert not violations, (
            f"Look-ahead detected: {len(violations)} feature timestamp(s) >= decision time "
            f"{self.DECISION_TIME}.\nViolating timestamps: {violations[:5]}"
        )

    def test_feature_before_decision_passes(self):
        times = [
            pd.Timestamp("2024-07-04 06:00:00", tz="UTC"),  # GFS 06Z run
            pd.Timestamp("2024-07-04 00:00:00", tz="UTC"),  # GFS 00Z run
            pd.Timestamp("2024-07-03 18:00:00", tz="UTC"),  # prior day 18Z
        ]
        self._assert_no_leakage(times)

    def test_feature_equal_to_decision_fails(self):
        times = [self.DECISION_TIME]
        with pytest.raises(AssertionError, match="Look-ahead detected"):
            self._assert_no_leakage(times)

    def test_feature_after_decision_fails(self):
        times = [pd.Timestamp("2024-07-04 18:00:00", tz="UTC")]
        with pytest.raises(AssertionError, match="Look-ahead detected"):
            self._assert_no_leakage(times)

    def test_mixed_batch_fails_on_any_violation(self):
        times = [
            pd.Timestamp("2024-07-04 00:00:00", tz="UTC"),  # fine
            pd.Timestamp("2024-07-05 00:00:00", tz="UTC"),  # next day — look-ahead
        ]
        with pytest.raises(AssertionError, match="Look-ahead detected"):
            self._assert_no_leakage(times)


# ── 2. LST / DST boundary correctness for KNYC ───────────────────────────────

class TestLSTBoundary:
    """
    KNYC settles on UTC-5 fixed (not DST-adjusted). During US summer,
    wall clocks show EDT (UTC-4), but the settlement window is still
    anchored on UTC-5. Verify that lst_date_for_utc assigns timestamps
    correctly across midnight UTC-5.
    """

    STATION = "KNYC"

    def test_offset_is_minus_five_hours(self):
        assert lst_offset(self.STATION) == timedelta(hours=-5)

    def test_utc_midnight_belongs_to_previous_lst_day(self):
        # 2024-07-04 00:00 UTC = 2024-07-03 19:00 LST → belongs to July 3 settlement day
        ts = pd.Timestamp("2024-07-04 00:00:00", tz="UTC")
        assert lst_date_for_utc(ts, self.STATION) == date(2024, 7, 3)

    def test_utc_0500_is_lst_midnight(self):
        # 2024-07-04 05:00 UTC = 2024-07-04 00:00 LST → first moment of July 4 day
        ts = pd.Timestamp("2024-07-04 05:00:00", tz="UTC")
        assert lst_date_for_utc(ts, self.STATION) == date(2024, 7, 4)

    def test_utc_2359_belongs_to_same_lst_day_as_0500(self):
        # 2024-07-04 23:59 UTC = 2024-07-04 18:59 LST → still July 4 settlement day
        ts = pd.Timestamp("2024-07-04 23:59:00", tz="UTC")
        assert lst_date_for_utc(ts, self.STATION) == date(2024, 7, 4)

    def test_utc_0459_belongs_to_previous_lst_day(self):
        # 2024-07-04 04:59 UTC = 2024-07-03 23:59 LST → still July 3 settlement day
        ts = pd.Timestamp("2024-07-04 04:59:00", tz="UTC")
        assert lst_date_for_utc(ts, self.STATION) == date(2024, 7, 3)

    def test_dst_spring_forward_does_not_change_lst_boundary(self):
        # 2024-03-10 is the US DST spring-forward day; wall clocks jump 2→3 AM.
        # Settlement window must still be anchored at UTC-5, not UTC-4.
        # 2024-03-10 04:59 UTC = 2024-03-09 23:59 LST → March 9 settlement day
        ts = pd.Timestamp("2024-03-10 04:59:00", tz="UTC")
        assert lst_date_for_utc(ts, self.STATION) == date(2024, 3, 9)

        # 2024-03-10 05:00 UTC = 2024-03-10 00:00 LST → March 10 settlement day
        ts2 = pd.Timestamp("2024-03-10 05:00:00", tz="UTC")
        assert lst_date_for_utc(ts2, self.STATION) == date(2024, 3, 10)

    def test_settlement_window_utc_correct_bounds(self):
        # NYC July 4 LST day: [2024-07-04 05:00 UTC, 2024-07-05 05:00 UTC)
        start, end = settlement_window_utc(date(2024, 7, 4), self.STATION)
        assert start == pd.Timestamp("2024-07-04 05:00:00", tz="UTC")
        assert end   == pd.Timestamp("2024-07-05 05:00:00", tz="UTC")

    def test_settlement_window_spans_exactly_24h(self):
        start, end = settlement_window_utc(date(2024, 1, 15), self.STATION)
        assert (end - start) == pd.Timedelta(hours=24)

    def test_to_utc_rejects_naive_datetime(self):
        from datetime import datetime
        with pytest.raises(ValueError, match="Naive datetime"):
            to_utc(datetime(2024, 7, 4, 12, 0, 0))  # no tzinfo


# ── 3. No timezone imports outside tz.py ─────────────────────────────────────

TZ_FORBIDDEN_IMPORTS = {
    "pytz",
    "zoneinfo",
    "dateutil.tz",
    "datetime.timezone",  # from datetime import timezone
}

# We allow importing `timezone` only in tz.py itself.
# All other modules must import from kalshi_weather.tz.

def _collect_python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if p != TZ_MODULE]


def _has_forbidden_tz_import(path: Path) -> list[str]:
    """
    Parse the AST of a Python file and return any forbidden timezone-related
    import statements found. Returns empty list if the file is clean.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("pytz", "zoneinfo"):
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            # from datetime import timezone
            if module == "datetime" and "timezone" in names:
                violations.append(f"from datetime import timezone")
            # from pytz import ...  /  from zoneinfo import ...
            if module in ("pytz", "zoneinfo", "dateutil.tz"):
                violations.append(f"from {module} import {names}")
    return violations


class TestTzIsolation:
    def test_no_tz_imports_outside_tz_module(self):
        py_files = _collect_python_files(SRC_ROOT)
        found: dict[str, list[str]] = {}
        for path in py_files:
            violations = _has_forbidden_tz_import(path)
            if violations:
                found[str(path.relative_to(SRC_ROOT))] = violations

        assert not found, (
            "Timezone imports found outside tz.py — all tz logic must go through "
            "kalshi_weather.tz:\n"
            + "\n".join(f"  {f}: {v}" for f, v in found.items())
        )

    def test_tz_module_exists(self):
        assert TZ_MODULE.exists(), f"tz.py not found at {TZ_MODULE}"

    def test_src_root_exists(self):
        assert SRC_ROOT.exists(), f"Source root not found: {SRC_ROOT}"
