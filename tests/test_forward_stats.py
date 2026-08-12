"""Forward-validation feedback loop: grade_candidates.forward_stats() and the
vet's _forward_record() surface. The graduation flag is ADVISORY — these tests
also pin that nothing here can silently promote a combo (the bar requires all
four conditions, and the vet keeps the validation check a FLAG)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import candidate_pipeline as cp
import grade_candidates as gc


def _df(rows):
    return pd.DataFrame(rows)


def _row(city="SAT", st="between", d="BUY_NO", date="2026-07-01", ret=0.4):
    return {"city": city, "strike_type": st, "direction": d,
            "settlement_date": date, "ret_per_dollar": ret}


def test_forward_stats_groups_by_city_type_direction():
    df = _df([_row(), _row(date="2026-07-02", ret=-0.2),
              _row(st="greater", date="2026-07-01", ret=0.1)])
    combos = gc.forward_stats(df)["combos"]
    assert combos["SAT|B|BUY_NO"]["n"] == 2
    assert combos["SAT|B|BUY_NO"]["days"] == 2
    assert combos["SAT|B|BUY_NO"]["mean_ret"] == 0.1
    assert combos["SAT|B|BUY_NO"]["win"] == 0.5
    assert combos["SAT|T|BUY_NO"]["n"] == 1


def test_graduation_needs_all_four_conditions():
    # 30 rows over 15 distinct days, all winners → meets the bar…
    win_rows = [_row(date=f"2026-06-{(i % 15) + 1:02d}", ret=0.2) for i in range(30)]
    assert gc.forward_stats(_df(win_rows))["combos"]["SAT|B|BUY_NO"]["graduated"]
    # …but not with too few distinct DAYS (same-day rows are one weather system),
    same_day = [_row(date="2026-06-01", ret=0.2) for _ in range(30)]
    assert not gc.forward_stats(_df(same_day))["combos"]["SAT|B|BUY_NO"]["graduated"]
    # …not with n below the bar,
    few = [_row(date=f"2026-06-{i+1:02d}", ret=0.2) for i in range(15)]
    assert not gc.forward_stats(_df(few))["combos"]["SAT|B|BUY_NO"]["graduated"]
    # …and not with a sub-bar win rate even when the mean is positive.
    mixed = [_row(date=f"2026-06-{(i % 15) + 1:02d}",
                  ret=(1.5 if i % 2 else -0.2)) for i in range(30)]
    c = gc.forward_stats(_df(mixed))["combos"]["SAT|B|BUY_NO"]
    assert c["mean_ret"] > 0 and not c["graduated"]


def test_forward_record_reads_artifact_and_formats(tmp_path, monkeypatch):
    doc = {"combos": {"SAT|B|BUY_NO": {"n": 41, "days": 18, "mean_ret": 0.31,
                                       "win": 0.88, "graduated": True}}}
    p = tmp_path / "forward_stats.json"
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(cp, "FWD_STATS", p)
    s = cp._forward_record("SAT", "B", "BUY_NO")
    assert "n=41/18d" in s and "win 88%" in s and "graduation bar" in s
    assert cp._forward_record("SAT", "T", "BUY_NO") is None


def test_forward_record_missing_artifact_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "FWD_STATS", tmp_path / "nope.json")
    assert cp._forward_record("SAT", "B", "BUY_NO") is None
