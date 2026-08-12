"""Price-path pilot (Steps 3-5): engine correctness + dormancy guardrails.

Synthetic panels only — no real data, no network. The guardrail tests pin the
property that makes this package safe to keep in the tree: it can never place
an order or touch the exchange."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kalshi_weather.backtest import maker_fee, taker_fee
from kalshi_weather.pricepath.engine import (
    POLICIES, day_block_ci, gated_entries, simulate, trade_pnl,
)

SRC = Path(__file__).parents[1] / "src" / "kalshi_weather" / "pricepath"
SCRIPTS = Path(__file__).parents[1] / "scripts"


def _panel(mids, ticker="KXHIGHNY-26AUG13-T96", prob=0.60, start="2026-08-13 12:00",
           spread=0.02, div_sign=1):
    """Synthetic one-ticker panel: mid path `mids` at 10-min spacing, constant
    spread, a fresh model view on every snapshot."""
    ts = pd.date_range(start, periods=len(mids), freq="10min", tz="UTC")
    mids = np.asarray(mids, dtype=float)
    bid, ask = mids - spread / 2, mids + spread / 2
    rt = spread + pd.Series(ask).map(taker_fee) + pd.Series(bid).map(taker_fee)
    return pd.DataFrame({
        "ticker": ticker, "snapshot_utc": ts,
        "yes_bid": bid, "yes_ask": ask, "yes_mid": mids,
        "spread": spread, "crossed": False,
        "has_model": True, "signal_age_min": 5.0,
        "prob_estimate": prob, "divergence": div_sign * (prob - mids),
        "settlement_date": "2026-08-13",
        "round_trip_taker": rt.to_numpy(),
    })


# ── entries ───────────────────────────────────────────────────────────────────

def test_gated_entry_is_first_qualifying_snapshot_only():
    p = _panel([0.30, 0.31, 0.32, 0.33], prob=0.60)
    e = gated_entries(p)
    assert len(e) == 1                       # one per (ticker, dir, day)
    assert e.iloc[0]["snapshot_utc"] == p["snapshot_utc"].iloc[0]
    assert e.iloc[0]["direction"] == "BUY_YES"


def test_small_divergence_does_not_enter():
    p = _panel([0.30] * 4, prob=0.32)        # 2c edge < ~2.8c round trip
    assert gated_entries(p).empty


# ── exits + pnl ───────────────────────────────────────────────────────────────

def test_fixed_horizon_exit_and_pessimistic_math():
    p = _panel([0.30, 0.33, 0.36, 0.39, 0.42], prob=0.60)
    t = simulate(p, POLICIES["fixed_30m"]).iloc[0]
    # entry at t0, exit at the first snapshot >= entry+30m = index 3 (0.39 mid)
    assert t["exit_utc"].endswith("12:30:00+00:00")
    a0, be = 0.30 + 0.01, 0.39 - 0.01
    assert t["pnl_pess"] == pytest.approx(be - a0 - taker_fee(a0) - taker_fee(be))
    assert t["pnl_opt"] == pytest.approx(
        0.39 - 0.30 - maker_fee(0.30) - maker_fee(0.39))


def test_buy_no_side_math():
    p = _panel([0.70, 0.60, 0.50], prob=0.30)    # model below market -> BUY_NO
    e = gated_entries(p)
    assert e.iloc[0]["direction"] == "BUY_NO"
    t = simulate(p, POLICIES["hold_to_end"]).iloc[0]
    b0, ae = 0.70 - 0.01, 0.50 + 0.01
    assert t["pnl_pess"] == pytest.approx(
        b0 - ae - taker_fee(1 - b0) - taker_fee(1 - ae))
    assert t["pnl_pess"] > 0                     # NO side won on this path


def test_target_model_exits_at_convergence():
    p = _panel([0.30, 0.40, 0.50, 0.60, 0.70], prob=0.55)
    t = simulate(p, POLICIES["target_model"]).iloc[0]
    assert t["exit_reason"] == "target_model"
    assert t["exit_utc"].endswith("12:30:00+00:00")   # first mid >= 0.55


def test_trailing_peak_exits_on_retrace():
    p = _panel([0.30, 0.40, 0.45, 0.41, 0.30], prob=0.60)
    t = simulate(p, POLICIES["trail_3c"]).iloc[0]
    assert t["exit_reason"] == "trail_3c"
    assert t["exit_utc"].endswith("12:30:00+00:00")   # 0.41 = 4c off the 0.45 peak


def test_untriggered_policy_exits_at_end_of_book():
    p = _panel([0.30, 0.31, 0.32], prob=0.60)
    t = simulate(p, POLICIES["target_model"]).iloc[0]  # never converges
    assert t["exit_reason"] == "end_of_book"
    assert t["exit_utc"] == p["snapshot_utc"].iloc[-1].isoformat()


def test_no_lookahead_exit_strictly_after_entry():
    p = _panel([0.30, 0.35], prob=0.60)
    t = simulate(p, POLICIES["hold_to_end"]).iloc[0]
    assert t["exit_utc"] > t["entry_utc"]
    # a ticker whose entry is its LAST print produces no trade at all
    p1 = _panel([0.30], prob=0.60)
    assert simulate(p1, POLICIES["hold_to_end"]).empty


def test_day_block_ci_shape():
    df = pd.DataFrame({"settlement_date": ["a"] * 3 + ["b"] * 3,
                       "pnl": [0.01, 0.02, 0.03, -0.01, 0.0, 0.01]})
    mean, lo, hi, n, k = day_block_ci(df, "pnl", n_boot=500)
    assert n == 6 and k == 2 and lo <= mean <= hi


# ── shadow book (Step 5) ──────────────────────────────────────────────────────

def _fake_logger_day(ob_dir: Path, d: date, mids):
    ts = pd.date_range(f"{d} 12:00", periods=len(mids), freq="10min", tz="UTC")
    rows = []
    for t, m in zip(ts, mids):
        rows.append({"snapshot_utc": t, "ticker": "KXHIGHNY-26AUG13-T96",
                     "side": "yes", "price_cents": int(m * 100 - 1), "quantity": 50})
        rows.append({"snapshot_utc": t, "ticker": "KXHIGHNY-26AUG13-T96",
                     "side": "no", "price_cents": int(100 - m * 100 - 1),
                     "quantity": 50})
    ob_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ob_dir / f"{d.isoformat()}.parquet", index=False)


def test_shadow_processes_day_and_is_idempotent(tmp_path, monkeypatch):
    import price_path_shadow as sh

    d = date.today() - timedelta(days=2)
    ob = tmp_path / "orderbook"
    _fake_logger_day(ob, d, [0.30, 0.40, 0.50, 0.60])
    _fake_logger_day(ob, d + timedelta(days=1), [0.65, 0.70])
    sig = tmp_path / "signals.jsonl"
    sig.write_text(json.dumps({
        "ticker": "KXHIGHNY-26AUG13-T96", "run_ts": f"{d}T11:55:00+00:00",
        "prob_estimate": 0.60, "settlement_date": "2026-08-13",
        "market_mid": 0.30, "direction": "BUY_YES"}) + "\n")

    out = tmp_path / "out"
    monkeypatch.setattr(sh, "OB_DIR", ob)
    monkeypatch.setattr(sh, "OUT_DIR", out)
    monkeypatch.setattr(sh, "STATE", out / "shadow_state.json")
    monkeypatch.setattr(sh, "LOG", out / "shadow_log.jsonl")
    import build_price_path_panel as bpp
    monkeypatch.setattr(bpp, "SIGNALS", sig)

    msg = sh.run(backfill=True)
    assert "appended" in msg
    rows = [json.loads(l) for l in (out / "shadow_log.jsonl").read_text().splitlines()]
    assert len(rows) == len(POLICIES)            # one ticker x every policy
    assert {r["mode"] for r in rows} == {"replay"}

    n_before = len(rows)
    sh.run(backfill=True)                        # idempotent re-run
    rows = [json.loads(l) for l in (out / "shadow_log.jsonl").read_text().splitlines()]
    assert len(rows) == n_before

    rep = sh.report(include_replay=True)
    assert "target_model" in rep and "UNPROVEN" in rep


# ── dormancy guardrails (the property that makes this safe to keep) ──────────

def test_pricepath_never_touches_exchange_or_network():
    sources = list(SRC.glob("*.py")) + [
        SCRIPTS / "price_path_shadow.py", SCRIPTS / "price_path_exit_bakeoff.py"]
    for f in sources:
        text = f.read_text()
        for forbidden in ("live.exchange", "live.executor", "import requests",
                          "create_order", "cancel_order", "KalshiExchange"):
            assert forbidden not in text, f"{f.name} references {forbidden!r}"


def test_pricepath_has_no_activation_flag():
    # Activation must be a conscious wiring change, not an env var someone can
    # flip: the package must not read ANY environment variable.
    for f in SRC.glob("*.py"):
        assert "getenv" not in f.read_text() and "environ" not in f.read_text()
