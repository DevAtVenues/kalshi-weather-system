"""Tests for scripts/reconcile_pnl.py — the claimed-vs-realized ledger.

The point of these is the DECOMPOSITION IDENTITY. A reconciliation tool whose
parts don't sum is worse than none, because it looks authoritative while
misattributing the loss. Every test here builds a scenario with a known answer
and checks the tool recovers it exactly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("reconcile_pnl", ROOT / "scripts" / "reconcile_pnl.py")
rec = importlib.util.module_from_spec(spec)
sys.modules["reconcile_pnl"] = rec
spec.loader.exec_module(rec)

T0 = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _fill(ticker, side, count, price_c, *, taker=True, at=T0, action="buy"):
    f = {"ticker": ticker, "side": side, "count": count, "action": action,
         "is_taker": taker, "created_time": at.isoformat()}
    f["yes_price" if side == "yes" else "no_price"] = price_c
    return f


def _settle(ticker, revenue_c, at=T0 + timedelta(days=1)):
    return {"ticker": ticker, "revenue": revenue_c, "settled_time": at.isoformat()}


def _signal(ticker, mid, prob, *, at=T0 - timedelta(hours=1), city="NYC",
            sdate="2026-07-16"):
    return {"ticker": ticker, "run_ts": at.isoformat(), "market_mid": mid,
            "prob_estimate": prob, "city": city, "settlement_date": sdate,
            "direction": "BUY_YES", "edge_raw": abs(prob - mid)}


def _ledger(fills, settlements, signals):
    by_ticker = {}
    for s in signals:
        by_ticker.setdefault(s["ticker"], []).append(s)
    rows = rec.build_positions(fills, settlements, by_ticker)
    return rows, rec.summarise(rows)


# ── the identity ────────────────────────────────────────────────────────────

def test_gap_decomposes_exactly_on_a_winner():
    """Bought YES at 57.5c on a contract the system scored at mid 50c, p=0.62.
    It settled YES. Every component is known by hand."""
    fills = [_fill("KXHIGHNY-A", "yes", 10, 58)]        # paid 58c
    setts = [_settle("KXHIGHNY-A", 1000)]               # 10 x 100c
    sigs = [_signal("KXHIGHNY-A", 0.50, 0.62)]
    rows, weeks = _ledger(fills, setts, sigs)
    r = rows[0]
    fee = rec.fee_cents(58, 10, True)

    assert r["cost_c"] == pytest.approx(580)
    assert r["realized_c"] == pytest.approx(1000 - 580 - fee)
    assert r["claimed_c"] == pytest.approx((100 - 50) * 10)      # outcome at mid
    assert r["expected_c"] == pytest.approx((0.62 * 100 - 50) * 10)
    assert r["slippage_c"] == pytest.approx((58 - 50) * 10)

    w = weeks["2026-W29"]
    assert w["gap_c"] == pytest.approx(w["model_error_c"] + w["slippage_c"] + w["fee_c"])


def test_gap_decomposes_exactly_on_a_loser():
    """Same identity must hold when the contract settles NO."""
    fills = [_fill("KXHIGHNY-B", "yes", 10, 58)]
    setts = [_settle("KXHIGHNY-B", 0)]
    sigs = [_signal("KXHIGHNY-B", 0.50, 0.62)]
    rows, weeks = _ledger(fills, setts, sigs)
    w = weeks["2026-W29"]
    assert w["gap_c"] == pytest.approx(w["model_error_c"] + w["slippage_c"] + w["fee_c"])
    assert rows[0]["claimed_c"] == pytest.approx(-50 * 10)


def test_identity_holds_across_a_mixed_book():
    """Winners, losers, both sides, maker and taker, several weeks."""
    fills, setts, sigs = [], [], []
    for i in range(12):
        tk = f"T{i}"
        side = "yes" if i % 2 else "no"
        price = 40 + i
        day = T0 + timedelta(days=i * 3)
        fills.append(_fill(tk, side, 5 + i, price, taker=(i % 3 != 0), at=day))
        setts.append(_settle(tk, (5 + i) * 100 if i % 4 else 0, at=day + timedelta(days=1)))
        sigs.append(_signal(tk, 0.30 + 0.03 * i, 0.35 + 0.03 * i,
                            at=day - timedelta(hours=2),
                            sdate=(day + timedelta(days=1)).date().isoformat()))
    rows, weeks = _ledger(fills, setts, sigs)
    assert len(rows) == 12
    for wk, w in weeks.items():
        assert w["gap_c"] == pytest.approx(
            w["model_error_c"] + w["slippage_c"] + w["fee_c"]), wk


# ── the join ────────────────────────────────────────────────────────────────

def test_uses_the_decision_that_stood_not_the_latest():
    """The whole point of joining on time: a later snapshot is not the decision
    the trade was made on. Grading against the last row is audit F34/F18."""
    fills = [_fill("TK", "yes", 1, 60, at=T0)]
    sigs = [_signal("TK", 0.50, 0.62, at=T0 - timedelta(hours=2)),   # stood
            _signal("TK", 0.90, 0.95, at=T0 + timedelta(hours=3))]   # after the fill
    rows, _ = _ledger(fills, [_settle("TK", 100)], sigs)
    assert rows[0]["scored_price_c"] == pytest.approx(50.0)
    assert rows[0]["prob_estimate"] == pytest.approx(0.62)


def test_no_signal_row_is_reported_not_silently_dropped():
    """A trade with no matching signal must still appear with realized P&L —
    a reconciliation tool that hides unmatched fills cannot find off-system trades."""
    rows, weeks = _ledger([_fill("GHOST", "yes", 4, 30)], [_settle("GHOST", 400)], [])
    assert len(rows) == 1
    assert rows[0]["matched"] is False
    assert rows[0]["claimed_c"] is None
    assert rows[0]["realized_c"] > 0
    assert sum(w["unmatched"] for w in weeks.values()) == 1


def test_no_signal_before_the_fill_does_not_match_a_later_one():
    fills = [_fill("TK", "yes", 1, 60, at=T0)]
    sigs = [_signal("TK", 0.9, 0.95, at=T0 + timedelta(hours=1))]
    rows, _ = _ledger(fills, [_settle("TK", 100)], sigs)
    assert rows[0]["matched"] is False


# ── the NO side ─────────────────────────────────────────────────────────────

def test_no_side_is_scored_against_the_no_price():
    """A BUY_NO scored against the YES mid would invert the whole ledger."""
    fills = [_fill("TK", "no", 10, 35)]
    sigs = [_signal("TK", 0.60, 0.40)]      # yes mid 60c -> no price 40c
    rows, _ = _ledger(fills, [_settle("TK", 1000)], sigs)
    r = rows[0]
    assert r["scored_price_c"] == pytest.approx(40.0)
    assert r["slippage_c"] == pytest.approx((35 - 40) * 10)          # bought BELOW mid
    assert r["expected_c"] == pytest.approx(((1 - 0.40) * 100 - 40) * 10)


# ── fees ────────────────────────────────────────────────────────────────────

def test_maker_fills_are_not_charged_and_takers_are():
    assert rec.fee_cents(50, 10, is_taker=False) == 0.0
    assert rec.fee_cents(50, 10, is_taker=True) > 0.0


def test_fee_peaks_at_the_midpoint():
    """0.07*p*(1-p) is maximal at p=0.5 — a cheap guard against an inverted formula."""
    mid = rec.fee_cents(50, 100, True)
    assert mid > rec.fee_cents(10, 100, True)
    assert mid > rec.fee_cents(90, 100, True)


# ── robustness ──────────────────────────────────────────────────────────────

def test_epoch_and_rfc3339_timestamps_both_parse():
    assert rec._ts(1752580800) is not None
    assert rec._ts("2026-07-15T12:00:00Z") is not None
    assert rec._ts("2026-07-15T12:00:00+00:00") is not None
    assert rec._ts(None) is None
    assert rec._ts("not a date") is None


def test_empty_inputs_do_not_crash():
    rows, weeks = _ledger([], [], [])
    assert rows == [] and weeks == {}
    assert "Nothing to reconcile" in rec.render(rows, weeks)


def test_sells_offset_buys_in_the_position():
    """Partial exits must reduce the position, not add to it."""
    fills = [_fill("TK", "yes", 10, 50), _fill("TK", "yes", 4, 60, action="sell")]
    rows, _ = _ledger(fills, [_settle("TK", 600)], [_signal("TK", 0.50, 0.60)])
    assert rows[0]["count"] == 6


def test_read_only_client_exposes_no_write_methods():
    """The interlock in live/exchange.py guards ORDER PLACEMENT. This client is
    safe without it only so long as it stays read-only."""
    forbidden = {"create_order", "cancel_order", "post", "place", "batch_create_orders"}
    assert forbidden.isdisjoint(dir(rec.ReadOnlyKalshi))
