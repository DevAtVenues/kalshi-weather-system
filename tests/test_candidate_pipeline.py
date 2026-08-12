"""Candidate pipeline: verdict mapping, settlement conventions, and the
partial-today-label guard (a label row for an unfinished day is not settlement
truth and must never reach the regime check or verdict grading)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import candidate_pipeline as cp


# ── settlement conventions (proven on settled outcomes) ──────────────────────

def test_greater_is_strictly_greater():
    # Canonical (settled-market derived): greater YES iff high > floor.
    assert cp._settles_yes("greater", 87.0, None, 87.0) is False   # == floor -> NO
    assert cp._settles_yes("greater", 87.0, None, 88.0) is True
    assert cp._settles_yes("greater", 87.0, None, 87.6) is True    # rounds to 88


def test_between_cap_inclusive():
    # Brackets are TWO integers wide, inclusive both ends ({84, 85}).
    assert cp._settles_yes("between", 84.0, 85.0, 84.0) is True
    assert cp._settles_yes("between", 84.0, 85.0, 85.0) is True
    assert cp._settles_yes("between", 84.0, 85.0, 86.0) is False
    assert cp._settles_yes("less", None, 91.0, 90.0) is True
    assert cp._settles_yes("less", None, 91.0, 91.0) is False


# ── verdict mapping ──────────────────────────────────────────────────────────

def _c(sev, name):
    return (sev, name, "note")


def test_fatal_red_is_pass():
    for name in ("meta", "mean_in_bucket", "longshot", "implausible"):
        assert cp.verdict_from([_c("RED", name)], 0.20) == "PASS"


def test_one_conditional_red_is_watch_two_are_pass():
    assert cp.verdict_from([_c("RED", "same_day")], 0.20) == "WATCH"
    assert cp.verdict_from([_c("RED", "model_spread")], 0.20) == "WATCH"
    assert cp.verdict_from([_c("RED", "same_day"), _c("RED", "model_spread")],
                           0.20) == "PASS"


def test_btype_yes_watch_is_conditional_never_trade_small():
    # Mirror of run_live Gate 4 (P1.7): the vet must never out-vote the push gate.
    assert cp.verdict_from([_c("RED", "btype_yes_watch")], 0.20) == "WATCH"
    assert cp.verdict_from([_c("RED", "btype_yes_watch"), _c("RED", "same_day")],
                           0.20) == "PASS"


def test_ceiling_flag_caps_at_watch():
    assert cp.verdict_from([_c("FLAG", "ceiling")], 0.20) == "WATCH"


def test_clean_candidate_trades_small():
    assert cp.verdict_from([_c("OK", "meta"), _c("OK", "models")], 0.10) == "TRADE_SMALL"
    assert cp.verdict_from([_c("FLAG", "validation")], 0.10) == "TRADE_SMALL"
    assert cp.verdict_from([_c("FLAG", "validation"), _c("FLAG", "regime")],
                           0.10) == "WATCH"   # 2 flags
    assert cp.verdict_from([], 0.05) == "WATCH"  # edge below the small-trade bar


# ── partial today-label guard (leakage) ──────────────────────────────────────

def test_labels_drop_unfinished_today(tmp_path, monkeypatch):
    today = date.today()
    df = pd.DataFrame({
        "date": [today - timedelta(days=2), today - timedelta(days=1), today],
        "high": [95.0, 94.0, 74.0],   # today's 74 = morning-so-far, NOT settlement
    })
    (tmp_path / "KDEN").mkdir()
    df.to_parquet(tmp_path / "KDEN" / "2026.parquet", index=False)
    monkeypatch.setattr(cp, "LABELS", tmp_path)
    out = cp._labels("KDEN")
    assert len(out) == 2
    assert out["date"].max() == pd.Timestamp(today - timedelta(days=1))
