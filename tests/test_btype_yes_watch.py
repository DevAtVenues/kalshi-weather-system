"""P1.7 — B-type BUY_YES watch-only gate.

The exclusive-cap fix (P1.6) removed fabricated YES edge on bracket contracts, but
the corrected BUY_YES direction is unproven forward (historical 5% win rate). A
B-type BUY_YES that clears every push gate must stay WATCH-ONLY — logged and graded,
never pushed — until >=10 graded post-fix actionable outcomes realize >=15% YES.

These tests pin the watch helper's accounting so the gate can't silently open early.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_RUN_LIVE = Path(__file__).parents[1] / "scripts" / "run_live.py"


@pytest.fixture()
def rl(monkeypatch, tmp_path):
    """Load scripts/run_live.py fresh with OUTCOMES_PATH pointed at a temp file."""
    spec = importlib.util.spec_from_file_location("run_live_under_test", _RUN_LIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "signal_outcomes.jsonl"
    monkeypatch.setattr(mod, "OUTCOMES_PATH", out)
    # Isolate ALL files push_forecast_picks writes so the test can never touch
    # production data (the integration cases call the real function, which appends
    # gate decisions to GATE_LOG and may persist notify state).
    monkeypatch.setattr(mod, "GATE_LOG", tmp_path / "gate_log.jsonl")
    monkeypatch.setattr(mod, "NOTIFY_STATE", tmp_path / "notified_today.json")
    mod._OUT = out  # test handle to write rows
    return mod


def _row(**kw):
    """A graded, actionable, post-fix ensemble B-type BUY_YES outcome by default."""
    base = {
        "ticker": "KXHIGHNY-26JUL15-B89.5", "run_ts": "2026-07-15T01:00:00+00:00",
        "prob_source": "ensemble", "strike_type": "between", "direction": "BUY_YES",
        "actionable": True, "settlement_date": "2026-07-15", "yes_settled": False,
    }
    base.update(kw)
    return base


def _write(mod, rows):
    mod._OUT.write_text("\n".join(json.dumps(r) for r in rows))


def test_no_outcomes_watch_active(rl):
    # No file / no rows → watch active, nothing to lift on.
    active, n, rate = rl._btype_yes_watch()
    assert active is True and n == 0 and rate == 0.0


def test_watch_active_below_min_n(rl):
    # 9 graded outcomes, all winners — still under the 10-outcome floor → stay active.
    _write(rl, [_row(ticker=f"T{i}-26JUL15-B89.5", yes_settled=True) for i in range(9)])
    active, n, rate = rl._btype_yes_watch()
    assert n == 9 and rate == 1.0 and active is True


def test_watch_lifts_at_min_n_and_rate(rl):
    # 10 graded, 3 YES (30% >= 15%) → both conditions met → watch lifts.
    rows = [_row(ticker=f"T{i}-26JUL15-B89.5", yes_settled=(i < 3)) for i in range(10)]
    _write(rl, rows)
    active, n, rate = rl._btype_yes_watch()
    assert n == 10 and rate == 0.3 and active is False


def test_watch_stays_active_when_rate_too_low(rl):
    # 10 graded but only 1 YES (10% < 15%) → the fix didn't prove out → stay WATCH.
    rows = [_row(ticker=f"T{i}-26JUL15-B89.5", yes_settled=(i < 1)) for i in range(10)]
    _write(rl, rows)
    active, n, rate = rl._btype_yes_watch()
    assert n == 10 and rate == 0.1 and active is True


def test_excludes_pre_fix_outcomes(rl):
    # Outcomes settled before the fix date must not count as forward evidence.
    rows = [_row(ticker=f"T{i}-26JUL05-B89.5", settlement_date="2026-07-05",
                 yes_settled=True) for i in range(10)]
    _write(rl, rows)
    active, n, rate = rl._btype_yes_watch()
    assert n == 0 and active is True


def test_excludes_non_actionable_and_wrong_type(rl):
    rows = [
        _row(ticker="A-26JUL15-B89.5", actionable=False, yes_settled=True),   # not actionable
        _row(ticker="B-26JUL15-T89",   strike_type="greater", yes_settled=True),  # T-type
        _row(ticker="C-26JUL15-B89.5", direction="BUY_NO",   yes_settled=True),   # BUY_NO
        _row(ticker="D-26JUL15-B89.5", prob_source="rule",   yes_settled=True),   # not ensemble
    ]
    _write(rl, rows)
    active, n, rate = rl._btype_yes_watch()
    assert n == 0 and active is True


def test_excludes_ungraded(rl):
    # A future/ungraded pick (no yes_settled) is not counted.
    rows = [_row(ticker="U-26JUL20-B89.5", settlement_date="2026-07-20")]
    for r in rows:
        r.pop("yes_settled")
    _write(rl, rows)
    active, n, rate = rl._btype_yes_watch()
    assert n == 0 and active is True


def test_dedup_latest_run_ts_per_ticker(rl):
    # Same ticker re-priced across runs counts once, using the LATEST run_ts snapshot.
    # Early snapshot says NO, final snapshot says YES — only the final should count.
    rows = [
        _row(ticker="X-26JUL15-B89.5", run_ts="2026-07-15T01:00:00+00:00", yes_settled=False),
        _row(ticker="X-26JUL15-B89.5", run_ts="2026-07-15T13:00:00+00:00", yes_settled=True),
    ]
    _write(rl, rows)
    active, n, rate = rl._btype_yes_watch()
    assert n == 1 and rate == 1.0


# ── Integration: a would-fire B-type BUY_YES is held by the gate, not pushed ──

def _fire_ready_signal():
    """One-row signals frame for NYC B89.5 BUY_YES that clears every upstream gate
    (liquid, edge>bar, near-term) so the ONLY thing that can stop it is the P1.7 gate."""
    import pandas as pd
    return pd.DataFrame([{
        "ticker": "KXHIGHNY-26JUL15-B89.5", "city": "NYC", "direction": "BUY_YES",
        "t_direction": "between", "settlement_date": "2026-07-15", "prob_source": "ensemble",
        "edge_raw": 0.15, "prob_estimate": 0.30, "market_mid": 0.15,
        "ens_p50": 89.1, "ens_sd": 1.4, "nws_disagree": 0.5, "is_same_day": False,
        "hours_to_settle": 20.0, "running_high": None, "tmax_f_fcst": 89.0,
        "pre_settlement": True, "yes_bid_dollars": 0.14, "yes_ask_dollars": 0.16,
        "spread": 0.02,
    }])


def _stub_upstream(rl, monkeypatch, *, watch_active):
    """Neutralize the network/tz/model helpers so the row reaches the P1.7 gate,
    and record any push attempt instead of sending it."""
    monkeypatch.setattr(rl, "_model_spread_f", lambda cfg, sd: 1.0)   # models agree
    monkeypatch.setattr(rl, "_hours_to_settlement", lambda sd, st: 20.0)
    monkeypatch.setattr(rl, "_btype_yes_watch", lambda: (watch_active, 0, 0.0))
    pushes = []
    monkeypatch.setattr(rl, "notify_transition",
                        lambda pick, old_cat=None: (pushes.append(pick), True)[1])
    return pushes


def test_would_fire_btype_yes_is_held_when_watch_active(rl, monkeypatch):
    pushes = _stub_upstream(rl, monkeypatch, watch_active=True)
    state = {"date": "2026-07-15", "picks": {}}          # fresh day → dedup can't save us
    pushed, filtered = rl.push_forecast_picks(_fire_ready_signal(), state)
    assert pushed == 0, "watch-active B-type BUY_YES must NOT push"
    assert pushes == [], "no notification should be attempted"
    assert "KXHIGHNY-26JUL15-B89.5:BUY_YES" not in state["picks"]


def test_same_signal_fires_once_watch_clears(rl, monkeypatch):
    # Control: with the watch lifted, the identical row DOES push — proving the P1.7
    # gate (not some other gate) is what held it above.
    pushes = _stub_upstream(rl, monkeypatch, watch_active=False)
    state = {"date": "2026-07-15", "picks": {}}
    pushed, filtered = rl.push_forecast_picks(_fire_ready_signal(), state)
    assert pushed == 1 and len(pushes) == 1
    assert state["picks"]["KXHIGHNY-26JUL15-B89.5:BUY_YES"] == "model_edge"
