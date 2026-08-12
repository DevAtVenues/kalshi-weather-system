"""Stale-fight fields (log-only): prob-age streaks, adverse-move sign per
direction, and the flag's thresholds. The MSP 2026-07-20 evening pattern —
static model + market drifting against us = growing fake edge — must trip it;
a fresh prob change or a favorable move must not."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import run_live as rl

NOW = datetime(2026, 7, 21, 4, 0, tzinfo=timezone.utc)


def _hist_line(ts, ticker, direction, prob, mid):
    return json.dumps({"run_ts": ts.isoformat(), "ticker": ticker,
                       "direction": direction, "prob_estimate": prob,
                       "market_mid": mid})


def _log(tmp_path, lines):
    p = tmp_path / "signals_log.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _sig(ticker, direction, prob, mid):
    return pd.DataFrame([{"ticker": ticker, "direction": direction,
                          "prob_estimate": prob, "market_mid": mid}])


def test_buy_no_adverse_is_mid_rising(tmp_path):
    # Model static 3h at 0.30 while NO-side edge grew because mid ROSE 0.08.
    lines = [_hist_line(NOW - timedelta(hours=h), "T1", "BUY_NO", 0.30, 0.50 + (3 - h) * 0.04)
             for h in (3, 2, 1)]
    st = rl._staleness_fields(_sig("T1", "BUY_NO", 0.30, 0.58),
                              log_path=_log(tmp_path, lines), now=NOW)
    f = st[("T1", "BUY_NO")]
    assert f["prob_age_min"] == 180.0
    assert f["mkt_move_adverse"] == 0.08
    assert f["stale_fight"] is True


def test_buy_yes_adverse_is_mid_falling(tmp_path):
    lines = [_hist_line(NOW - timedelta(hours=3), "T2", "BUY_YES", 0.70, 0.60)]
    st = rl._staleness_fields(_sig("T2", "BUY_YES", 0.70, 0.52),
                              log_path=_log(tmp_path, lines), now=NOW)
    f = st[("T2", "BUY_YES")]
    assert f["mkt_move_adverse"] == 0.08 and f["stale_fight"] is True
    # favorable move (mid rising toward us) must NOT trip it
    st = rl._staleness_fields(_sig("T2", "BUY_YES", 0.70, 0.68),
                              log_path=_log(tmp_path, lines), now=NOW)
    assert st[("T2", "BUY_YES")]["stale_fight"] is False


def test_prob_change_resets_streak(tmp_path):
    # Old reads at 0.30, latest logged read at 0.42 → current 0.42 streak is
    # only 30 min old: fresh signal, not a stale fight.
    lines = [_hist_line(NOW - timedelta(hours=4), "T3", "BUY_NO", 0.30, 0.50),
             _hist_line(NOW - timedelta(minutes=30), "T3", "BUY_NO", 0.42, 0.57)]
    st = rl._staleness_fields(_sig("T3", "BUY_NO", 0.42, 0.58),
                              log_path=_log(tmp_path, lines), now=NOW)
    f = st[("T3", "BUY_NO")]
    assert f["prob_age_min"] == 30.0 and f["stale_fight"] is False


def test_current_prob_differs_from_all_history_is_age_zero(tmp_path):
    lines = [_hist_line(NOW - timedelta(hours=2), "T4", "BUY_NO", 0.30, 0.50)]
    st = rl._staleness_fields(_sig("T4", "BUY_NO", 0.45, 0.55),
                              log_path=_log(tmp_path, lines), now=NOW)
    f = st[("T4", "BUY_NO")]
    assert f["prob_age_min"] == 0.0 and f["stale_fight"] is False


def test_no_history_is_unknown_not_fresh(tmp_path):
    st = rl._staleness_fields(_sig("T5", "BUY_NO", 0.30, 0.50),
                              log_path=_log(tmp_path, [_hist_line(
                                  NOW - timedelta(hours=1), "OTHER", "BUY_NO", 0.3, 0.5)]),
                              now=NOW)
    f = st[("T5", "BUY_NO")]
    assert f["prob_age_min"] is None and f["stale_fight"] is False


def test_short_or_small_moves_do_not_trip(tmp_path):
    # 90 min static (< 120) with a big move → no trip; 3h static with a 3¢
    # move (< 5¢) → no trip.
    lines = [_hist_line(NOW - timedelta(minutes=90), "T6", "BUY_NO", 0.30, 0.50)]
    st = rl._staleness_fields(_sig("T6", "BUY_NO", 0.30, 0.58),
                              log_path=_log(tmp_path, lines), now=NOW)
    assert st[("T6", "BUY_NO")]["stale_fight"] is False
    lines = [_hist_line(NOW - timedelta(hours=3), "T7", "BUY_NO", 0.30, 0.50)]
    st = rl._staleness_fields(_sig("T7", "BUY_NO", 0.30, 0.53),
                              log_path=_log(tmp_path, lines), now=NOW)
    assert st[("T7", "BUY_NO")]["stale_fight"] is False


def test_missing_log_returns_empty(tmp_path):
    st = rl._staleness_fields(_sig("T8", "BUY_NO", 0.3, 0.5),
                              log_path=tmp_path / "nope.jsonl", now=NOW)
    assert st == {}
