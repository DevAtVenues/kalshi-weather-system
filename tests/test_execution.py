"""Execution layer: signed exchange client safety + executor lifecycle.

No network anywhere: KalshiExchange is only exercised up to signing (real RSA
key generated per test), and the Executor runs against an in-memory fake
exchange with the risk layer redirected to tmp_path (same isolation pattern
as test_risk_wiring)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import kalshi_weather.live.executor as executor_mod
from kalshi_weather.live.exchange import ExchangeError, KalshiExchange
from kalshi_weather.live.executor import Executor, maybe_executor
from kalshi_weather.live.risk import RiskManager


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def rsa_key(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = tmp_path / "demo_key.pem"
    pem.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    return key, pem


@pytest.fixture()
def demo_env(monkeypatch, rsa_key):
    _, pem = rsa_key
    monkeypatch.delenv("KALSHI_EXEC_ENV", raising=False)
    monkeypatch.delenv("KALSHI_EXEC_ALLOW_PROD", raising=False)
    monkeypatch.setenv("KALSHI_DEMO_KEY_ID", "demo-key-id")
    monkeypatch.setenv("KALSHI_DEMO_PRIVATE_KEY_PATH", str(pem))


def _row(**kw) -> pd.Series:
    base = dict(ticker="KXHIGHNY-26AUG13-T96", direction="BUY_YES", city="NYC",
                settlement_date="2026-08-13", prob_estimate=0.45, market_mid=0.30,
                yes_bid_dollars=0.28, yes_ask_dollars=0.32, strike_type="greater",
                threshold_f=96.0, floor_strike=None, cap_strike=None,
                edge_raw=0.15)
    base.update(kw)
    return pd.Series(base)


class FakeExchange:
    env = "demo"

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.fills: list[dict] = []
        self.settlements: list[dict] = []
        self.canceled: list[str] = []
        self._n = 0

    def create_order(self, ticker, side, count, price_cents,
                     expiration_ts=None, client_order_id=None):
        self._n += 1
        oid = f"ord-{self._n}"
        self.orders[oid] = {"order_id": oid, "ticker": ticker, "side": side,
                            "count": count, "price_cents": price_cents,
                            "expiration_ts": expiration_ts, "status": "resting"}
        return self.orders[oid]

    def cancel_order(self, order_id):
        self.canceled.append(order_id)
        self.orders[order_id]["status"] = "canceled"
        return {}

    def get_order(self, order_id):
        return self.orders.get(order_id, {"status": "canceled"})

    def get_fills(self, min_ts=None):
        return list(self.fills)

    def get_settlements(self, limit=100):
        return list(self.settlements)


@pytest.fixture()
def rig(tmp_path):
    """(executor, fake_exchange, risk) with every store under tmp_path."""
    cfgp, stp = tmp_path / "risk_config.json", tmp_path / "risk_state.json"
    risk = RiskManager(config_path=cfgp, state_path=stp)
    fake = FakeExchange()
    ex = Executor(fake, risk,
                  state_path=tmp_path / "exec_state.json",
                  log_path=tmp_path / "exec_log.jsonl",
                  watchlist_path=tmp_path / "watchlist.json")
    return ex, fake, risk


def _log_events(ex: Executor) -> list[dict]:
    p = ex._log_path
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines()]


# ── exchange client safety ────────────────────────────────────────────────────

def test_demo_is_default_env(demo_env):
    ex = KalshiExchange()
    assert ex.env == "demo"
    assert "demo-api.kalshi.co" in ex.base


def test_prod_requires_interlock(demo_env, monkeypatch, rsa_key):
    _, pem = rsa_key
    monkeypatch.setenv("KALSHI_PROD_KEY_ID", "prod-key-id")
    monkeypatch.setenv("KALSHI_PROD_PRIVATE_KEY_PATH", str(pem))
    with pytest.raises(ExchangeError, match="refusing PROD"):
        KalshiExchange(env="prod")
    monkeypatch.setenv("KALSHI_EXEC_ALLOW_PROD",
                       "I_UNDERSTAND_THIS_TRADES_REAL_MONEY")
    ex = KalshiExchange(env="prod")
    assert "api.elections.kalshi.com" in ex.base


def test_missing_creds_raise(monkeypatch):
    monkeypatch.delenv("KALSHI_DEMO_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_DEMO_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(ExchangeError, match="KEY_ID"):
        KalshiExchange()


def test_signature_verifies(demo_env, rsa_key):
    key, _ = rsa_key
    ex = KalshiExchange()
    sig = base64.b64decode(ex._sign("1712345", "GET", "/portfolio/balance"))
    key.public_key().verify(         # raises InvalidSignature on mismatch
        sig, b"1712345GET/trade-api/v2/portfolio/balance",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())


def test_order_validation(demo_env):
    ex = KalshiExchange()
    with pytest.raises(ExchangeError, match="bad side"):
        ex.create_order("T", "maybe", 1, 50)
    with pytest.raises(ExchangeError, match="bad price"):
        ex.create_order("T", "yes", 1, 0)
    with pytest.raises(ExchangeError, match="bad count"):
        ex.create_order("T", "yes", 0, 50)


def test_maybe_executor_requires_opt_in(monkeypatch):
    monkeypatch.delenv("KALSHI_EXEC", raising=False)
    assert maybe_executor() is None


def test_maybe_executor_survives_misconfig(monkeypatch):
    monkeypatch.setenv("KALSHI_EXEC", "1")
    monkeypatch.delenv("KALSHI_DEMO_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_DEMO_PRIVATE_KEY_PATH", raising=False)
    assert maybe_executor() is None    # loud, but never kills the signal run


def test_maybe_executor_demo_uses_isolated_risk(monkeypatch, demo_env, tmp_path):
    monkeypatch.setenv("KALSHI_EXEC", "1")
    real_cfg = tmp_path / "real_risk_config.json"
    real_cfg.write_text(json.dumps({"bankroll": 3000.0}))
    monkeypatch.setattr(executor_mod, "_REAL_RISK_CONFIG", real_cfg)
    monkeypatch.setattr(executor_mod, "EXEC_DIR", tmp_path / "execution")
    monkeypatch.setattr(executor_mod, "DEMO_RISK_CONFIG",
                        tmp_path / "execution" / "demo_risk_config.json")
    monkeypatch.setattr(executor_mod, "DEMO_RISK_STATE",
                        tmp_path / "execution" / "demo_risk_state.json")
    monkeypatch.setattr(executor_mod, "STATE_PATH",
                        tmp_path / "execution" / "executor_state.json")
    monkeypatch.setattr(executor_mod, "LOG_PATH",
                        tmp_path / "execution" / "execution_log.jsonl")
    ex = maybe_executor()
    assert ex is not None and ex.exchange.env == "demo"
    # demo risk config was seeded from the real one, but lives separately
    assert ex.risk.cfg["bankroll"] == 3000.0
    assert (tmp_path / "execution" / "demo_risk_config.json").exists()


# ── executor lifecycle ────────────────────────────────────────────────────────

def test_buy_yes_places_maker_at_bid(rig):
    ex, fake, _ = rig
    out = ex.execute_pick(_row())
    assert out is not None
    order = fake.orders[out["order_id"]]
    assert order["side"] == "yes"
    assert order["price_cents"] == 28          # joins the bid, never crosses
    assert order["count"] >= 1
    assert order["expiration_ts"] is not None  # server-side TTL auto-exit
    assert ex.state["orders"][out["order_id"]]["ticker"] == "KXHIGHNY-26AUG13-T96"
    assert [e["event"] for e in _log_events(ex)] == ["order_placed"]


def test_buy_no_price_is_complement_of_ask(rig):
    ex, fake, _ = rig
    out = ex.execute_pick(_row(direction="BUY_NO", prob_estimate=0.10))
    order = fake.orders[out["order_id"]]
    assert order["side"] == "no"
    assert order["price_cents"] == 100 - 32


def test_duplicate_pick_not_reordered(rig):
    ex, fake, _ = rig
    ex.execute_pick(_row())
    ex.execute_pick(_row())
    assert len(fake.orders) == 1
    assert _log_events(ex)[-1]["event"] == "skip_duplicate"


def test_negative_kelly_is_skipped(rig):
    ex, fake, _ = rig
    assert ex.execute_pick(_row(prob_estimate=0.20)) is None   # model below mid
    assert not fake.orders
    assert _log_events(ex)[-1]["event"] == "skip_sizing"


def test_halted_risk_places_nothing(rig):
    ex, fake, risk = rig
    risk.record_pnl(-1e9)          # trip the breaker
    assert risk.is_halted()
    assert ex.execute_pick(_row()) is None
    assert not fake.orders
    assert _log_events(ex)[-1]["event"] == "skip_halted"


def test_fill_updates_risk_positions_and_watchlist(rig):
    from kalshi_weather.live.runner import CITY_CONFIGS
    ex, fake, risk = rig
    out = ex.execute_pick(_row())
    oid, count = out["order_id"], out["count"]
    fake.fills = [{"trade_id": "t1", "order_id": oid, "side": "yes",
                   "count": count, "yes_price": 28, "no_price": 72,
                   "created_time": "2026-08-12T14:00:00Z"}]
    ex.reconcile()

    pos = ex.state["positions"]["KXHIGHNY-26AUG13-T96"]
    assert pos["contracts"] == count
    assert pos["cost_dollars"] == pytest.approx(count * 0.28)
    assert oid not in ex.state["orders"]                    # fully filled
    assert risk.open_exposure()["NYC_2026-08-13"] == pytest.approx(count * 0.28)

    wl = json.loads(ex._watchlist_path.read_text())
    assert wl[0]["ticker"] == "KXHIGHNY-26AUG13-T96"
    assert wl[0]["direction"] == "BUY_YES"
    assert wl[0]["strike_type"] == "greater" and wl[0]["threshold"] == 96
    assert wl[0]["station"] == CITY_CONFIGS["NYC"]["nws_station"]

    ex.reconcile()                       # same fill again: idempotent
    assert ex.state["positions"]["KXHIGHNY-26AUG13-T96"]["contracts"] == count
    assert len(json.loads(ex._watchlist_path.read_text())) == 1


def test_settlement_records_pnl_and_win_math(rig):
    ex, fake, risk = rig
    ex.state["positions"]["KXHIGHNY-26AUG13-T96"] = {
        "direction": "BUY_YES", "side": "yes", "contracts": 10,
        "cost_dollars": 2.8, "city": "NYC", "settlement_date": "2026-08-13"}
    fake.settlements = [{"ticker": "KXHIGHNY-26AUG13-T96", "market_result": "yes"}]
    ex.reconcile()
    assert not ex.state["positions"]
    assert risk.daily_pnl() == pytest.approx(10 - 2.8)
    assert _log_events(ex)[-1]["pnl_dollars"] == pytest.approx(7.2)


def test_losing_settlement_trips_breaker_and_cancels_resting(rig, tmp_path):
    ex, fake, risk = rig
    risk.cfg["bankroll"] = 40.0        # breaker at -$2.80
    out = ex.execute_pick(_row())      # a resting order that must die on halt
    ex.state["positions"]["KXHIGHCHI-26AUG13-B83.5"] = {
        "direction": "BUY_NO", "side": "no", "contracts": 10,
        "cost_dollars": 5.0, "city": "CHI", "settlement_date": "2026-08-13"}
    fake.settlements = [{"ticker": "KXHIGHCHI-26AUG13-B83.5",
                         "market_result": "yes"}]     # our NO loses
    ex.reconcile()
    assert risk.daily_pnl() == pytest.approx(-5.0)
    assert risk.is_halted()
    assert fake.canceled == [out["order_id"]]
    assert not ex.state["orders"]


def test_expired_order_dropped_from_state(rig):
    ex, fake, _ = rig
    out = ex.execute_pick(_row())
    fake.orders[out["order_id"]]["status"] = "expired"
    ex.reconcile()
    assert not ex.state["orders"]
    assert _log_events(ex)[-1]["event"] == "order_gone"
