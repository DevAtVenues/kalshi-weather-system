"""Skill monitor: hygiene rules and alarm logic (each guards a known artifact)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import skill_monitor as sm


def _df(rows):
    return pd.DataFrame(rows)


def _row(ticker="T1", run_ts="2026-07-01T00:00:00", h=24.0, prob=0.7, mid=0.5,
         yes=True, day="2026-07-02", source="ensemble"):
    return {"ticker": ticker, "run_ts": run_ts, "hours_to_settle": h,
            "prob_estimate": prob, "market_mid": mid, "yes_settled": yes,
            "settlement_date": day, "prob_source": source}


def test_resolved_market_rows_excluded(tmp_path, monkeypatch):
    """h<=0 rows (mid≈0.99, market already resolved) must not flatter the baseline."""
    f = tmp_path / "o.jsonl"
    import json
    rows = [_row(), _row(ticker="T2", h=-3.0, mid=0.99)]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(sm, "OUTCOMES", f)
    df = sm.load_rows()
    assert set(df["ticker"]) == {"T1"}


def test_one_row_per_ticker_and_lead_last_read_wins(tmp_path, monkeypatch):
    import json
    rows = [_row(run_ts="2026-07-01T00:00:00", prob=0.6),
            _row(run_ts="2026-07-01T06:00:00", prob=0.8),      # later read
            _row(run_ts="2026-07-01T06:00:00", h=10.0, prob=0.9)]  # same-day bucket
    f = tmp_path / "o.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(sm, "OUTCOMES", f)
    df = sm.load_rows()
    ahead = df[df["lead"] == "day-ahead"]
    assert len(ahead) == 1 and float(ahead["prob_estimate"].iloc[0]) == 0.8
    assert len(df[df["lead"] == "same-day"]) == 1


def _analysis(skill, ci, bias=0.0, bias_ci=(-0.01, 0.01), days=14):
    return {"source": "ensemble", "lead": "day-ahead", "n": 200, "days": days,
            "brier_model": 0.1, "brier_market": 0.1 + skill,
            "skill": skill, "skill_ci": ci, "bias": bias, "bias_ci": bias_ci}


def test_alarm_red_when_market_significantly_better():
    s, _ = sm.status_of(_analysis(-0.02, (-0.03, -0.01)))
    assert s == "RED"


def test_warn_when_negative_but_ci_spans_zero():
    s, _ = sm.status_of(_analysis(-0.01, (-0.02, 0.005)))
    assert s == "WARN"


def test_red_on_significant_bias_even_with_ok_skill():
    s, why = sm.status_of(_analysis(0.01, (0.001, 0.02), bias=0.08,
                                    bias_ci=(0.03, 0.13)))
    assert s == "RED" and "over-prediction" in why


def test_warn_on_thin_effective_n():
    s, _ = sm.status_of(_analysis(0.01, (0.001, 0.02), days=4))
    assert s == "WARN"


def test_ok_when_beating_market():
    s, _ = sm.status_of(_analysis(0.02, (0.005, 0.035)))
    assert s == "OK"


def test_bootstrap_deterministic():
    df = _df([_row(ticker=f"T{i}", day=f"2026-07-{d:02d}")
              for i in range(20) for d in (1, 2, 3)])
    df["y"] = 1.0
    df["brier_model"] = (df["prob_estimate"] - df["y"]) ** 2
    df["brier_market"] = (df["market_mid"] - df["y"]) ** 2
    fn = lambda d: float((d["brier_market"] - d["brier_model"]).mean())
    assert sm.block_bootstrap_ci(df, fn, iters=50) == sm.block_bootstrap_ci(df, fn, iters=50)
