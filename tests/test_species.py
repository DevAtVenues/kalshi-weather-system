"""Species labels: deterministic mapping from board-row fields to the
hypothesized edge source. The labels partition the graded record into
separately-validatable strategies (structural short-bracket vs front-running
fresh divergence vs other)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import candidate_pipeline as cp


def _r(**kw):
    base = {"strike_type": "greater", "direction": "BUY_NO",
            "floor_strike": None, "cap_strike": None,
            "prob_age_min": None, "stale_fight": None}
    base.update(kw)
    return pd.Series(base)


def test_narrow_bracket_buy_no_is_structural():
    r = _r(strike_type="between", floor_strike=98.0, cap_strike=99.0)
    assert cp.species_of(r) == "short_bracket_no"
    # …even when the read is fresh: structural label wins (it's the strategy)
    r = _r(strike_type="between", floor_strike=98.0, cap_strike=99.0,
           prob_age_min=10.0)
    assert cp.species_of(r) == "short_bracket_no"


def test_wide_bracket_or_buy_yes_is_not_structural():
    r = _r(strike_type="between", floor_strike=95.0, cap_strike=97.0,
           prob_age_min=10.0)
    assert cp.species_of(r) == "fresh_divergence"     # 2°F bracket ≠ narrow
    r = _r(strike_type="between", direction="BUY_YES",
           floor_strike=98.0, cap_strike=99.0)
    assert cp.species_of(r) != "short_bracket_no"


def test_fresh_divergence_needs_recent_prob_and_no_stale_fight():
    assert cp.species_of(_r(prob_age_min=30.0)) == "fresh_divergence"
    assert cp.species_of(_r(prob_age_min=300.0)) == "model_edge_other"
    assert cp.species_of(_r(prob_age_min=30.0, stale_fight=True)) == "model_edge_other"
    assert cp.species_of(_r(prob_age_min=None)) == "model_edge_other"


def test_nan_fields_are_safe():
    r = _r(strike_type="between", floor_strike=float("nan"), cap_strike=99.0,
           prob_age_min=float("nan"), stale_fight=float("nan"))
    assert cp.species_of(r) == "model_edge_other"
