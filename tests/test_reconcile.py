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


def _settle(ticker, revenue_c, at=T0 + timedelta(days=1), result="yes"):
    return {"ticker": ticker, "revenue": revenue_c, "market_result": result,
            "settled_time": at.isoformat()}


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
    setts = [_settle("KXHIGHNY-A", 1000)]               # 10 x 100c, YES won
    sigs = [_signal("KXHIGHNY-A", 0.50, 0.62)]
    rows, weeks = _ledger(fills, setts, sigs)
    r = rows[0]
    fee = rec.model_fee_cents(58, 10, True)

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
    setts = [_settle("KXHIGHNY-B", 0, result="no")]     # we held YES -> lost
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
        setts.append(_settle(tk, (5 + i) * 100 if i % 4 else 0,
                             at=day + timedelta(days=1),
                             result=(side if i % 4 else ("no" if side == "yes" else "yes"))))
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
    w = list(weeks.values())[0]
    assert w["n_unattributed"] == 1
    assert w["unattributed_realized_c"] == pytest.approx(rows[0]["realized_c"])
    # ...but it must NOT enter the attributed decomposition, or the identity breaks.
    assert w["n_attributed"] == 0
    assert w["realized_c"] == pytest.approx(0.0)
    assert w["realized_total_c"] == pytest.approx(rows[0]["realized_c"])


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
    rows, _ = _ledger(fills, [_settle("TK", 1000, result="no")], sigs)
    r = rows[0]
    assert r["scored_price_c"] == pytest.approx(40.0)
    assert r["slippage_c"] == pytest.approx((35 - 40) * 10)          # bought BELOW mid
    assert r["expected_c"] == pytest.approx(((1 - 0.40) * 100 - 40) * 10)


# ── fees ────────────────────────────────────────────────────────────────────

def test_maker_fee_follows_the_repo_model_not_zero():
    """The live path submits post-only MAKER orders. Charging them zero would
    report no fees at all for the strategy's usual fill (Codex P1)."""
    taker = rec.model_fee_cents(50, 10, is_taker=True)
    maker = rec.model_fee_cents(50, 10, is_taker=False)
    assert 0 < maker < taker
    assert maker == pytest.approx(taker * rec.MAKER_FEE_RATIO, abs=1.0)


def test_fee_peaks_at_the_midpoint():
    """0.07*p*(1-p) is maximal at p=0.5 — a cheap guard against an inverted formula."""
    mid = rec.model_fee_cents(50, 100, True)
    assert mid > rec.model_fee_cents(10, 100, True)
    assert mid > rec.model_fee_cents(90, 100, True)


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


def test_render_covers_every_bucket_without_raising():
    """render() reads keys summarise() produces; a schema change that drops one
    only surfaces here, not in the arithmetic tests."""
    fills = [_fill("A", "yes", 5, 60, at=T0),                 # attributed winner
             _fill("B", "yes", 5, 60, at=T0),                 # attributed loser
             _fill("OPEN", "yes", 5, 60, at=T0),              # never settles
             _fill("MANUAL", "yes", 5, 20, at=T0)]            # no signal
    setts = [_settle("A", 500), _settle("B", 0, result="no"), _settle("MANUAL", 500)]
    sigs = [_signal("A", 0.5, 0.6), _signal("B", 0.5, 0.6), _signal("OPEN", 0.5, 0.6)]
    rows, weeks = _ledger(fills, setts, sigs)
    out = rec.render(rows, weeks)
    for token in ("expect$", "claim$", "real$", "TOTAL",
                  "still open", "unattributed", "Gap attribution"):
        assert token in out, token


def test_sells_offset_buys_in_the_position():
    """Partial exits must reduce the position, not add to it."""
    fills = [_fill("TK", "yes", 10, 50), _fill("TK", "yes", 4, 60, action="sell")]
    rows, _ = _ledger(fills, [_settle("TK", 600)], [_signal("TK", 0.50, 0.60)])
    assert rows[0]["count"] == 6


# ── regressions on the Codex review of PR #1 ────────────────────────────────

def test_unsettled_position_is_not_booked_as_a_total_loss():
    """P1. A fill with no settlement yet is OPEN. Treating its payout as zero
    records the whole cost as a realized loss and corrupts the current week."""
    rows, weeks = _ledger([_fill("OPEN", "yes", 10, 40)], [], [_signal("OPEN", 0.35, 0.50)])
    r = rows[0]
    assert r["settled"] is False
    assert r["realized_c"] is None and r["payout_c"] is None
    w = list(weeks.values())[0]
    assert w["n_open"] == 1
    assert w["realized_c"] == pytest.approx(0.0)
    assert w["open_cost_c"] > 0


def test_open_and_settled_positions_in_one_week_do_not_contaminate():
    fills = [_fill("DONE", "yes", 10, 40), _fill("OPEN", "yes", 10, 40)]
    sigs = [_signal("DONE", 0.35, 0.50), _signal("OPEN", 0.35, 0.50)]
    rows, weeks = _ledger(fills, [_settle("DONE", 1000)], sigs)
    w = list(weeks.values())[0]
    assert (w["n_attributed"], w["n_open"]) == (1, 1)
    assert w["gap_c"] == pytest.approx(w["model_error_c"] + w["slippage_c"] + w["fee_c"])


def test_exchange_reported_fee_beats_the_model():
    """P1. A reconciliation tool must prefer what was actually charged."""
    f = _fill("TK", "yes", 10, 50)
    f["fee_cents"] = 37.0
    rows, _ = _ledger([f], [_settle("TK", 1000)], [_signal("TK", 0.50, 0.60)])
    assert rows[0]["fee_c"] == pytest.approx(37.0)
    assert rows[0]["fee_source"] == "api"


def test_dollar_denominated_fee_field_is_scaled_to_cents():
    f = _fill("TK", "yes", 10, 50)
    f["fee"] = 0.42                       # dollars, not cents
    rows, _ = _ledger([f], [_settle("TK", 1000)], [_signal("TK", 0.50, 0.60)])
    assert rows[0]["fee_c"] == pytest.approx(42.0)


def test_falls_back_to_the_model_when_no_fee_is_reported():
    rows, _ = _ledger([_fill("TK", "yes", 10, 50)], [_settle("TK", 1000)],
                      [_signal("TK", 0.50, 0.60)])
    assert rows[0]["fee_source"] == "model"
    assert rows[0]["fee_c"] == pytest.approx(rec.model_fee_cents(50, 10, True))


def test_both_sides_of_one_ticker_do_not_each_claim_the_full_payout():
    """P2. Ticker-wide revenue copied into a YES row and a NO row double-counts
    the payout and marks both sides winners."""
    fills = [_fill("TK", "yes", 10, 60), _fill("TK", "no", 10, 40)]
    sigs = [_signal("TK", 0.55, 0.60)]
    rows, _ = _ledger(fills, [_settle("TK", 1000, result="yes")], sigs)
    by_side = {r["side"]: r for r in rows}
    assert by_side["yes"]["won"] is True and by_side["yes"]["payout_c"] == pytest.approx(1000)
    assert by_side["no"]["won"] is False and by_side["no"]["payout_c"] == pytest.approx(0.0)
    assert by_side["yes"]["realized_c"] > 0 > by_side["no"]["realized_c"]


def test_each_fill_joins_to_its_own_contemporaneous_signal():
    """P2. A position accumulated across cycles must price its later fills against
    the later mid, not the mid that stood at the first fill."""
    fills = [_fill("TK", "yes", 10, 50, at=T0),
             _fill("TK", "yes", 10, 80, at=T0 + timedelta(hours=6))]
    sigs = [_signal("TK", 0.50, 0.60, at=T0 - timedelta(hours=1)),
            _signal("TK", 0.75, 0.85, at=T0 + timedelta(hours=5))]
    rows, _ = _ledger(fills, [_settle("TK", 2000)], sigs)
    r = rows[0]
    # scored notional = 50*10 + 75*10 = 1250c over 20 contracts
    assert r["scored_price_c"] == pytest.approx(62.5)
    assert r["slippage_c"] == pytest.approx((50 - 50) * 10 + (80 - 75) * 10)
    assert r["expected_c"] == pytest.approx((60 - 50) * 10 + (85 - 75) * 10)
    assert r["n_fills"] == 2 and r["n_matched_fills"] == 2


def test_partially_matched_position_is_not_attributed():
    """One fill before any signal existed: the position's attribution would be
    built on a decision that did not cover every contract."""
    fills = [_fill("TK", "yes", 10, 50, at=T0 - timedelta(hours=3)),   # pre-signal
             _fill("TK", "yes", 10, 50, at=T0)]
    sigs = [_signal("TK", 0.50, 0.60, at=T0 - timedelta(hours=1))]
    rows, weeks = _ledger(fills, [_settle("TK", 2000)], sigs)
    assert rows[0]["matched"] is False and rows[0]["expected_c"] is None
    w = list(weeks.values())[0]
    assert w["n_attributed"] == 0 and w["n_unattributed"] == 1


def test_weekly_identity_survives_a_book_with_open_and_unattributed_rows():
    """The whole point of the buckets: the printed decomposition must still add up."""
    fills, setts, sigs = [], [], []
    for i in range(9):
        tk, day = f"M{i}", T0 + timedelta(days=i)
        fills.append(_fill(tk, "yes", 5, 45 + i, at=day))
        sigs.append(_signal(tk, 0.40, 0.55, at=day - timedelta(hours=1),
                            sdate=(day + timedelta(days=1)).date().isoformat()))
    for i in (0, 1, 2, 3, 4, 5):                       # 6 settle, 3 stay open
        setts.append(_settle(f"M{i}", 500 if i % 2 else 0,
                             at=T0 + timedelta(days=i, hours=20),
                             result="yes" if i % 2 else "no"))
    fills.append(_fill("MANUAL", "yes", 3, 20, at=T0))  # no signal at all
    setts.append(_settle("MANUAL", 300))
    rows, weeks = _ledger(fills, setts, sigs)
    assert sum(w["n_open"] for w in weeks.values()) == 3
    assert sum(w["n_unattributed"] for w in weeks.values()) == 1
    for wk, w in weeks.items():
        assert w["gap_c"] == pytest.approx(
            w["model_error_c"] + w["slippage_c"] + w["fee_c"]), wk


def test_read_only_client_exposes_no_write_methods():
    """The interlock in live/exchange.py guards ORDER PLACEMENT. This client is
    safe without it only so long as it stays read-only."""
    forbidden = {"create_order", "cancel_order", "post", "place", "batch_create_orders"}
    assert forbidden.isdisjoint(dir(rec.ReadOnlyKalshi))
