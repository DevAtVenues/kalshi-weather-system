"""Settlement conventions vs REALITY — the anti-circularity gate.

The fixture is 2,052 settled Kalshi markets (every boundary case in the
2026-05→07 history — high == floor/cap for between, within 1°F of the strike
for greater/less — plus a bulk sample) joined with official CLI highs.
kalshi_weather.settlement must reproduce Kalshi's actual result for EVERY row.

This exists because the previous conventions ("between excludes cap", "greater
is >= floor") were validated only against our own labels + our own convention —
circular — and misgraded 10.5% of between markets and every high==floor greater.
A hand audit is not evidence; settled markets are. If Kalshi ever changes rules,
this file is the tripwire (refresh the fixture via the harvest in
data/analysis/settled_convention_harvest.json regeneration).
"""
import json
from pathlib import Path

import pytest

from kalshi_weather.settlement import settles_yes, yes_bounds

FIXTURE = Path(__file__).parent / "fixtures" / "settlement_convention.json"


def _rows():
    return json.loads(FIXTURE.read_text())


def test_fixture_present_and_substantial():
    rows = _rows()
    assert len(rows) > 1500
    assert {r["type"] for r in rows} == {"between", "greater", "less"}


def test_every_settled_market_reproduced():
    bad = []
    for r in _rows():
        got = settles_yes(r["type"], r["floor"], r["cap"], r["high"])
        want = r["result"] == "yes"
        if got is None or got != want:
            bad.append(r)
    assert not bad, f"{len(bad)} settled markets misgraded; first: {bad[:3]}"


def test_boundary_cases_covered():
    rows = _rows()
    b_at_cap = [r for r in rows if r["type"] == "between" and r["high"] == r["cap"]]
    g_at_floor = [r for r in rows if r["type"] == "greater" and r["high"] == r["floor"]]
    l_at_cap = [r for r in rows if r["type"] == "less" and r["high"] == r["cap"]]
    assert len(b_at_cap) >= 100 and all(r["result"] == "yes" for r in b_at_cap)
    assert len(g_at_floor) >= 20 and all(r["result"] == "no" for r in g_at_floor)
    assert len(l_at_cap) >= 20 and all(r["result"] == "no" for r in l_at_cap)


class TestRules:
    def test_between_inclusive_both_ends(self):
        assert settles_yes("between", 101, 102, 101) is True
        assert settles_yes("between", 101, 102, 102) is True
        assert settles_yes("between", 101, 102, 100) is False
        assert settles_yes("between", 101, 102, 103) is False

    def test_greater_strict(self):
        assert settles_yes("greater", 96, None, 96) is False
        assert settles_yes("greater", 96, None, 97) is True

    def test_less_strict(self):
        assert settles_yes("less", None, 89, 89) is False
        assert settles_yes("less", None, 89, 88) is True

    def test_unknown_type(self):
        assert settles_yes("weird", 1, 2, 1) is None


class TestBounds:
    def test_bounds_match_discrete_rule(self):
        # every integer high inside yes_bounds must settle YES and vice versa
        for st, fl, cp in (("between", 101, 102), ("greater", 96, None),
                           ("less", None, 89)):
            lo, hi = yes_bounds(st, fl, cp)
            for h in range(80, 115):
                inside = lo <= h < hi
                assert settles_yes(st, fl, cp, h) == inside, (st, h)

    def test_between_width_is_two_integers(self):
        lo, hi = yes_bounds("between", 101, 102)
        assert lo == pytest.approx(100.5) and hi == pytest.approx(102.5)
