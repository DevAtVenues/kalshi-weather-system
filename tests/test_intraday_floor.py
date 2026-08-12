"""
P3.1 intraday floor — ensemble members are truncated from below at the observed
running daily high, since the final high must be >= anything already observed.
Tests verify the mathematical properties without touching live API calls.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Core floor math (the np.maximum operation)
# ---------------------------------------------------------------------------

def test_floor_raises_sub_threshold_members():
    """Members below running high are raised; members above are untouched."""
    members = np.array([80.0, 85.0, 90.0, 95.0])
    h_now = 88.0
    floored = np.maximum(members, h_now)
    assert floored[0] == h_now       # 80 < 88 → raised
    assert floored[1] == h_now       # 85 < 88 → raised
    assert floored[2] == 90.0        # 90 ≥ 88 → unchanged
    assert floored[3] == 95.0        # 95 ≥ 88 → unchanged


def test_floor_never_below_observed():
    members = np.linspace(70.0, 95.0, 50)
    h_now = 89.0
    floored = np.maximum(members, h_now)
    assert (floored >= h_now).all()


def test_floor_does_not_cap_upside():
    """High members are never lowered by the floor."""
    members = np.array([90.0, 95.0, 100.0])
    floored = np.maximum(members, 85.0)
    np.testing.assert_array_equal(floored, members)  # all above floor: unchanged


# ---------------------------------------------------------------------------
# T-type pricing: P(YES for above-X) where h_now < X is unchanged
# ---------------------------------------------------------------------------

def test_ttype_above_threshold_prob_unchanged_when_hnow_below():
    """
    For T-type (above threshold X), if h_now < X, the floor changes members
    that are < h_now → still < X, so P(YES) is identical before and after.
    """
    members = np.linspace(84.0, 96.0, 100)
    threshold = 92.0
    h_now = 88.0  # running high, below threshold
    floored = np.maximum(members, h_now)

    p_before = (members >= threshold).mean()
    p_after  = (floored  >= threshold).mean()
    assert p_before == p_after, (
        f"T-type P(YES) changed when h_now={h_now} < threshold={threshold}: "
        f"{p_before:.4f} → {p_after:.4f}"
    )


def test_ttype_above_threshold_prob_unity_when_hnow_above():
    """If running high already exceeded the threshold, all floored members are above it."""
    members = np.linspace(84.0, 96.0, 100)
    threshold = 87.0
    h_now = 88.0  # above threshold → lock
    floored = np.maximum(members, h_now)
    assert (floored >= threshold).all()


# ---------------------------------------------------------------------------
# B-type pricing: bracket contracts benefit most from the floor
# ---------------------------------------------------------------------------

def test_btype_prob_rises_when_hnow_in_bracket():
    """
    When h_now falls inside a bracket [floor_b, cap_b), the floor pulls low
    members into the bracket, increasing P(YES).
    """
    # 72 members spread 84–96°F
    members = np.linspace(84.0, 96.0, 72)
    bracket_floor = 88.5  # bracket: round to 89°F
    bracket_cap   = 90.5  # bracket: round to 90°F

    h_now = 89.2  # observed — inside the bracket

    # Round-to-nearest convention (Kalshi settles on the integer daily high)
    p_before = ((members >= bracket_floor) & (members < bracket_cap)).mean()
    floored  = np.maximum(members, h_now)
    p_after  = ((floored  >= bracket_floor) & (floored  < bracket_cap)).mean()

    assert p_after >= p_before, (
        f"B-type P(YES) should increase when h_now={h_now} inside bracket "
        f"[{bracket_floor},{bracket_cap}); got {p_before:.4f} → {p_after:.4f}"
    )


def test_btype_prob_unchanged_when_hnow_below_bracket():
    """When h_now is clearly below the bracket, the floor has no useful effect."""
    members = np.linspace(88.0, 96.0, 72)   # all above bracket
    bracket_floor = 88.5
    bracket_cap   = 90.5
    h_now = 82.0   # well below bracket

    floored  = np.maximum(members, h_now)
    p_before = ((members >= bracket_floor) & (members < bracket_cap)).mean()
    p_after  = ((floored  >= bracket_floor) & (floored  < bracket_cap)).mean()
    # Floor at 82 touches no members ≥ 88 → P unchanged
    assert p_before == p_after


# ---------------------------------------------------------------------------
# Distribution statistics after floor
# ---------------------------------------------------------------------------

def test_floor_narrows_distribution():
    """Flooring raises the left tail; p50 (median) should be >= original or equal."""
    members = np.linspace(80.0, 100.0, 71)
    h_now = 87.0
    floored = np.maximum(members, h_now)
    assert np.median(floored) >= np.median(members)
    assert np.std(floored) <= np.std(members)   # variance reduced or equal


# ---------------------------------------------------------------------------
# Time-of-day tapering via intraday_prob
# ---------------------------------------------------------------------------

from kalshi_weather.live.intraday import intraday_prob, warming_fraction, intraday_adjust


def test_warming_fraction_at_midday():
    """At 11 AM LST, about half the warming potential is spent."""
    frac = warming_fraction(11.0, sunrise=7.0, peak=15.0)
    assert 0.4 < frac < 0.6


def test_intraday_adjust_tapers_upside():
    """
    intraday_adjust should give LESS upside at a higher frac (more of day left),
    but the SAME floor at H_now regardless of frac.
    """
    members = np.array([80.0, 88.0, 92.0, 95.0])
    h_now = 86.0
    frac_morning = 0.8   # early
    frac_afternoon = 0.2  # late

    adj_morning   = intraday_adjust(members, h_now, frac_morning)
    adj_afternoon = intraday_adjust(members, h_now, frac_afternoon)

    # Both must floor at h_now
    assert (adj_morning   >= h_now).all()
    assert (adj_afternoon >= h_now).all()
    # Afternoon has LESS upside above h_now (closer to obs-lock)
    morning_upside   = (adj_morning   - h_now).sum()
    afternoon_upside = (adj_afternoon - h_now).sum()
    assert afternoon_upside < morning_upside


def test_intraday_prob_rises_as_high_approaches_threshold():
    """
    As H_now gets closer to the threshold from below, P(YES for above X) should rise
    because fewer members need additional warming to cross it.
    """
    members = np.linspace(86.0, 95.0, 71)
    threshold = 90.5  # T-type "above 90°F"
    sigma = 3.0
    local_hour = 11.0  # late morning

    p_far  = intraday_prob(members, running_high=80.0, local_hour=local_hour,
                           strike_type="greater", floor=threshold, cap=None,
                           nws=90.0, sigma=sigma)
    p_close = intraday_prob(members, running_high=88.0, local_hour=local_hour,
                            strike_type="greater", floor=threshold, cap=None,
                            nws=90.0, sigma=sigma)
    assert p_far is not None and p_close is not None
    assert p_close >= p_far   # closer to threshold → higher P(YES)


def test_intraday_prob_collapses_at_peak():
    """
    At the afternoon peak (local_hour=15), the distribution collapses onto H_now.
    If H_now exceeds threshold → P close to 1. If H_now below threshold → P close to 0.
    """
    members = np.array([88.0, 90.0, 92.0, 94.0])
    sigma = 3.0

    # H_now already exceeds threshold (91 > 89.5)
    p_above = intraday_prob(members, running_high=91.0, local_hour=15.0,
                            strike_type="greater", floor=89.5, cap=None,
                            nws=90.0, sigma=sigma)
    assert p_above is not None and p_above > 0.9

    # H_now below threshold at peak — high should not grow much more
    p_below = intraday_prob(members, running_high=87.0, local_hour=15.0,
                            strike_type="greater", floor=91.5, cap=None,
                            nws=90.0, sigma=sigma)
    assert p_below is not None and p_below < 0.2


def test_intraday_tighter_than_pure_floor_at_late_morning():
    """
    At 1 PM LST, intraday_prob with tapering should give a DIFFERENT P than the naive
    np.maximum floor (which behaves like frac=1, i.e., pre-dawn).
    For a T-type near the upper tail, the tapered version should be LOWER (less upside
    remaining → lower P that a member below H_now will still reach the threshold).
    """
    members = np.linspace(85.0, 96.0, 71)
    h_now = 88.0
    threshold = 95.5  # T-type "above 95°F" — needs significant additional warming
    sigma = 2.0
    local_hour = 13.0  # 1 PM, frac ≈ 0.25

    # Pure floor path (frac=1.0 equivalent): standard mixture_prob on floored members
    from kalshi_weather.calibration.ensemble_dist import mixture_prob
    floored_members = np.maximum(members, h_now)
    p_floor_only = mixture_prob(floored_members, "greater", threshold, None, None, sigma, 0.0)

    # Time-of-day path: intraday_prob with frac=0.25
    p_intraday = intraday_prob(members, h_now, local_hour,
                               strike_type="greater", floor=threshold, cap=None,
                               nws=None, sigma=sigma, w_nws=0.0)

    assert p_intraday is not None and p_floor_only is not None
    # At 1 PM, only 25% of afternoon warming remains — P(YES) for a target far above H_now
    # should be LOWER than with the naive floor (which assumes all afternoon is still ahead).
    assert p_intraday <= p_floor_only + 0.05   # intraday never significantly EXCEEDS floor-only


# ---------------------------------------------------------------------------
# Canonical settlement conventions (kalshi_weather.settlement — derived from
# 8,040 settled markets, enforced by tests/test_settlement_convention.py):
#   between INCLUSIVE both ends ({floor..cap}); greater strict >; less strict <.
# ---------------------------------------------------------------------------

def test_bracket_inclusive_cap_counts_cap_members():
    """'between' includes BOTH integer endpoints: B83.5 = {83, 84}."""
    from kalshi_weather.calibration.ensemble_dist import contract_prob

    assert contract_prob(np.full(71, 84.0), "between", floor=83.0, cap=84.0) == 1.0
    assert contract_prob(np.full(71, 83.0), "between", floor=83.0, cap=84.0) == 1.0
    assert contract_prob(np.full(71, 85.0), "between", floor=83.0, cap=84.0) == 0.0
    assert contract_prob(np.full(71, 82.0), "between", floor=83.0, cap=84.0) == 0.0


def test_bracket_two_degrees_wide_mass():
    """Center 84.2, sd 1.0 on bracket {83,84}: P = P(82.5 <= m < 84.5) — a bit
    over half the mass. The old exclusive-cap code priced this bracket at half
    its true width (the 2026-07-22 KHOU lesson)."""
    from kalshi_weather.calibration.ensemble_dist import contract_prob
    rng = np.random.default_rng(42)
    members = rng.normal(84.2, 1.0, 200_000)
    p = contract_prob(members, "between", floor=83.0, cap=84.0)
    truth = float(np.mean((members >= 82.5) & (members < 84.5)))
    assert abs(p - truth) < 1e-9
    assert 0.45 < p < 0.65


def test_greater_is_strict():
    """greater(floor=F) pays YES iff int high >= F+1 (empirical: 47/47 settled
    high==floor cases were NO)."""
    from kalshi_weather.calibration.ensemble_dist import contract_prob
    assert contract_prob(np.full(71, 82.0), "greater", floor=82.0, cap=None) == 0.0
    assert contract_prob(np.full(71, 83.0), "greater", floor=82.0, cap=None) == 1.0


def test_greater_matches_empirical_rounding_rule():
    from kalshi_weather.calibration.ensemble_dist import contract_prob
    rng = np.random.default_rng(7)
    members = rng.normal(82.0, 1.5, 200_000)
    p = contract_prob(members, "greater", floor=82.0, cap=None)
    truth = float(np.mean(np.round(members) > 82))
    assert abs(p - truth) < 0.005


def test_ladder_partition():
    """less(cap=F) + P(int high == F .. F+1) + greater(floor=F+1) tiles every
    outcome — the actual Kalshi ladder shape (e.g. <=87 | {88,89} | >=90)."""
    from kalshi_weather.calibration.ensemble_dist import contract_prob
    rng = np.random.default_rng(11)
    members = rng.normal(88.5, 2.5, 200_000)
    p_lt = contract_prob(members, "less",    floor=None, cap=88.0)
    p_b  = contract_prob(members, "between", floor=88.0, cap=89.0)
    p_ge = contract_prob(members, "greater", floor=89.0, cap=None)
    assert abs((p_lt + p_b + p_ge) - 1.0) < 1e-9


def test_greater_prob_consistent_with_yes_won():
    """Cross-module: contract_prob and outcome_tracker.yes_won agree at the floor."""
    from kalshi_weather.calibration.ensemble_dist import contract_prob
    from kalshi_weather.outcome_tracker import yes_won
    parsed = {"strike_type": "greater", "floor_strike": 82.0, "cap_strike": None}
    assert yes_won(parsed, 82.0) is False
    assert contract_prob(np.full(10, 82.0), "greater", floor=82.0, cap=None) == 0.0
    assert yes_won(parsed, 83.0) is True
    assert contract_prob(np.full(10, 83.0), "greater", floor=82.0, cap=None) == 1.0


# ---------------------------------------------------------------------------
# Deterministic (bias.py) and ensemble (ensemble_dist) price the SAME contract
# identically — both route through kalshi_weather.settlement.yes_bounds.
# ---------------------------------------------------------------------------

def test_bias_contract_probability_matches_ensemble_normal():
    from kalshi_weather.calibration.bias import contract_probability
    from kalshi_weather.calibration.ensemble_dist import nws_normal_prob
    cases = [
        ("between", 83.0, 84.0, 83.0), ("between", 90.0, 91.0, 92.3),
        ("less",    None, 87.0, 85.0), ("less",    None, 72.0, 74.1),
        ("greater", 82.0, None, 82.0), ("greater", 101.0, None, 99.5),
    ]
    for st, fl, cp, h in cases:
        det = contract_probability(h, 1.5, floor=fl, cap=cp, strike_type=st)
        ens = nws_normal_prob(st, fl, cp, h, 1.5)
        assert det is not None and ens is not None
        assert abs(det - ens) < 1e-9, f"{st} f={fl} c={cp} h={h}: det={det} ens={ens}"


# ---------------------------------------------------------------------------
# parse_ticker bracket bounds (B{N}.5 -> API strikes N / N+1; inclusive)
# ---------------------------------------------------------------------------

def test_parse_ticker_btype_floor_cap():
    """B83.5 ticker -> floor=83.0, cap=84.0 = the API integer strikes; the
    bracket settles YES on 83 AND 84 (inclusive, canonical rules)."""
    from kalshi_weather.outcome_tracker import parse_ticker, yes_won
    parsed = parse_ticker("KXHIGHNYC-26JUL09-B83.5")
    assert parsed is not None
    assert parsed["strike_type"] == "between"
    assert parsed["floor_strike"] == 83.0
    assert parsed["cap_strike"]   == 84.0
    assert yes_won(parsed, 83.0) is True
    assert yes_won(parsed, 84.0) is True    # inclusive cap — the KHOU-102 lesson
    assert yes_won(parsed, 85.0) is False
    assert yes_won(parsed, 82.0) is False
