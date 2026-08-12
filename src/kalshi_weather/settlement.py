"""
Canonical Kalshi high-temp settlement conventions — the ONLY place they live.

Derived EMPIRICALLY 2026-07-22 from 8,040 settled markets joined with official
CLI highs across all 20 series (fixture: tests/fixtures/settlement_convention.json,
enforced by tests/test_settlement_convention.py — the build fails if these rules
stop matching settled reality):

    between (floor F, cap C):  YES  iff  F <= high <= C     (INCLUSIVE both ends)
        - high == floor -> YES: 576/576 settled cases
        - high == cap   -> YES: 565/565 settled cases
        - every bracket is TWO integers wide ({F, F+1}); ladders step by 2
    greater (floor F):         YES  iff  high >  F  (i.e. high >= F+1)
        - high == floor -> NO: 47/47 settled cases
    less (cap C):              YES  iff  high <  C  (i.e. high <= C-1)
        - high == cap   -> NO: 97/97 settled cases

    The ladder tiles perfectly:  less(C) = <=C-1 | B(C).5 = {C, C+1} | ...
    ... | B(F-1).5 = {F-1, F} | greater(F) = >=F+1.

History: the codebase carried "between = [floor, cap) exclusive" from the first
backtest (misgrading 10.5% of all between markets — every high==cap case), and a
2026-07-10 "fix" changed greater to >= floor on a false empirical claim. Both
were self-consistently validated against our own labels+convention (circular),
and two days of live trades (SAT 98 in {98,99}, DFW 106 past {103,104}) happened
to settle identically under either reading. The first discriminating outcome
(KHOU 102 in a {101,102} bracket, 2026-07-22) exposed it. Hand audits are not
evidence; the fixture test is.

Continuous-space boundaries: CLI settles the integer high = round(true max), so
    between: F-0.5 <= t < C+0.5
    greater: t >= F+0.5
    less:    t <  C-0.5
"""
from __future__ import annotations

import math


def settles_yes(strike_type: str, floor: float | None, cap: float | None,
                high: float) -> bool | None:
    """YES/NO for a SETTLED integer high. None if the contract shape is unknown."""
    st = (strike_type or "").lower()
    h = int(round(high))
    if st == "between" and floor is not None and cap is not None:
        return int(floor) <= h <= int(cap)
    if st == "greater" and floor is not None:
        return h > int(floor)
    if st == "less" and cap is not None:
        return h < int(cap)
    return None


def yes_bounds(strike_type: str, floor: float | None,
               cap: float | None) -> tuple[float, float] | None:
    """[lo, hi) interval of the CONTINUOUS true max over which YES settles.
    Use for any distribution-based P(YES): P = P(lo <= t < hi)."""
    st = (strike_type or "").lower()
    if st == "between" and floor is not None and cap is not None:
        return (float(floor) - 0.5, float(cap) + 0.5)
    if st == "greater" and floor is not None:
        return (float(floor) + 0.5, math.inf)
    if st == "less" and cap is not None:
        return (-math.inf, float(cap) - 0.5)
    return None
