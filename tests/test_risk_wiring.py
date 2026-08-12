"""Risk layer wiring — real fills/P&L must reach the breaker, and a tripped
breaker must actually suppress pushes (CLAUDE.md rule 6: a bad day can never
cascade). These paths were dead code until 2026-07-09: nothing called
record_fill/record_pnl, so the circuit breaker could never trip."""
import json
from datetime import date

import kalshi_weather.live.risk as risk_mod
import kalshi_weather.dashboard.notifications as notif
from kalshi_weather.live.risk import RiskManager, city_from_market, parse_dollars


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(risk_mod, "CONFIG_PATH", tmp_path / "risk_config.json")
    monkeypatch.setattr(risk_mod, "STATE_PATH", tmp_path / "risk_state.json")


def test_parse_dollars_sheet_formats():
    assert parse_dollars("$3.71") == 3.71
    assert parse_dollars("($4.49)") == -4.49
    assert parse_dollars("$1,200.50") == 1200.50
    assert parse_dollars("-") is None
    assert parse_dollars("") is None
    assert parse_dollars(None) is None
    assert parse_dollars(2.5) == 2.5


def test_city_from_market_longest_match_wins():
    assert city_from_market("Miami High Temp, Jun 15 — Between 89.5-91.5°F") == "MIA"
    assert city_from_market("Highest Temp NYC, 6/16: 75°–76°") == "NYC"
    # "New Orleans" must not be shadowed by any shorter name
    assert city_from_market("New Orleans High Temp, Jul 8 — T91") == "MSY"
    assert city_from_market("Some Unknown Market") is None


def test_breaker_trips_and_persists_across_instances(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    rm = RiskManager()                     # default config: $250 bankroll, 7% CB = $17.50
    rm.record_pnl(-10.0)
    assert not rm.is_halted()
    rm.record_pnl(-10.0)                   # cumulative -$20 crosses the breaker
    assert rm.is_halted()
    assert RiskManager().is_halted()       # fresh instance reads the same state


def test_long_running_process_sees_cross_process_halt(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    engine = RiskManager()                 # e.g. run_live_engine, stays resident
    logger = RiskManager()                 # e.g. log_to_sheets, separate process
    logger.record_pnl(-999.0)
    assert not engine._state["halted"]     # stale in-memory view
    engine.reset_if_new_day()              # called at the top of every cycle
    assert engine.is_halted()


def test_halted_state_suppresses_pushes_today_only(monkeypatch, tmp_path):
    p = tmp_path / "risk_state.json"
    monkeypatch.setattr(notif, "_RISK_STATE", p)
    p.write_text(json.dumps({"date": date.today().isoformat(), "halted": True}))
    assert notif._risk_halted_today()
    p.write_text(json.dumps({"date": "2020-01-01", "halted": True}))   # stale day
    assert not notif._risk_halted_today()
    p.unlink()                                                          # no state file
    assert not notif._risk_halted_today()


def test_iso_day_normalizes_every_caller_format():
    import pandas as pd
    from datetime import datetime
    from kalshi_weather.live.risk import _iso_day
    assert _iso_day(date(2026, 6, 15)) == "2026-06-15"          # signal rows (date)
    assert _iso_day(datetime(2026, 6, 15, 12)) == "2026-06-15"  # datetime
    assert _iso_day(pd.Timestamp("2026-06-15")) == "2026-06-15" # pandas
    assert _iso_day("2026-06-15") == "2026-06-15"               # ISO string
    assert _iso_day("06/15/2026") == "2026-06-15"               # trade sheet format


def test_sheet_fill_binds_signal_corr_cap(monkeypatch, tmp_path):
    """A fill recorded from the trade sheet (MM/DD/YYYY) must reduce corr-cap
    headroom for a signal keyed by a date object — the two sources build the
    same exposure key. This was broken before _iso_day normalization."""
    _isolate(monkeypatch, tmp_path)
    import pandas as pd
    rm = RiskManager()                     # corr cap: 5% of $250 = $12.50
    rm.record_fill("MIA", "06/15/2026", contracts=20, dollars_at_risk=10.0)
    row = pd.Series({"prob_estimate": 0.70, "market_mid": 0.50, "direction": "BUY_YES",
                     "city": "MIA", "settlement_date": date(2026, 6, 15)})
    s = rm.size_trade(row)
    assert s["cap"] == "corr"              # $2.50 headroom binds before the $3.75 trade cap
    assert s["dollars"] <= 2.50 + 1e-9
    rm.record_fill("MIA", date(2026, 6, 15), 5, 2.50)   # fill from a signal-side caller
    assert rm.size_trade(row)["blocked"] == "corr_cap_full"


def test_notify_returns_false_only_on_halt(monkeypatch, tmp_path):
    p = tmp_path / "risk_state.json"
    monkeypatch.setattr(notif, "_RISK_STATE", p)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    pick = {"cat": "lock", "city": "MIA", "contract_label": "X", "action": "BUY NO"}
    p.write_text(json.dumps({"date": date.today().isoformat(), "halted": True}))
    assert notif.notify_transition(pick, None) is False           # halted → suppressed
    assert notif.notify_transition({"cat": "watchlist"}, None) is True  # never pushed anyway
    p.write_text(json.dumps({"date": date.today().isoformat(), "halted": False}))
    assert notif.notify_transition(pick, None) is True            # not halted, no topic → True


def test_update_risk_state_end_to_end(monkeypatch, tmp_path):
    """Trade-sheet rows drive the risk state: an Open fill lands under the right
    city-date exposure key; a big Settled loss trips the breaker."""
    _isolate(monkeypatch, tmp_path)
    import importlib.util
    from pathlib import Path
    script = Path(__file__).parents[1] / "scripts" / "log_to_sheets.py"
    spec = importlib.util.spec_from_file_location("lts", script)
    lts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lts)
    rm = RiskManager()
    trades = [
        {"date": "06/15/2026", "market": "Miami High Temp, Jun 15 — B91.5",
         "position": "No", "amount_committed": "$3.71", "entry_price": "$0.53",
         "status": "Open"},
        {"date": "06/15/2026", "market": "Highest Temp NYC, 6/15: 75°–76°",
         "net_pl": "($20.00)", "status": "Settled"},
    ]
    lts.update_risk_state(trades, rm)
    assert rm.open_exposure().get("MIA_2026-06-15") == 3.71
    assert rm.is_halted()                  # -$20 crosses the $17.50 default breaker


def test_sized_trade_still_respects_caps(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    import pandas as pd
    rm = RiskManager()
    row = pd.Series({"prob_estimate": 0.70, "market_mid": 0.50, "direction": "BUY_YES",
                     "city": "MIA", "settlement_date": "2026-07-09"})
    s = rm.size_trade(row)
    assert s["blocked"] is None
    # per-trade cap: 1.5% of $250 = $3.75 → at 50¢, at most 7 contracts
    assert s["contracts"] * 0.50 <= 0.015 * 250 + 1e-9
    rm.record_pnl(-999.0)
    assert rm.size_trade(row)["blocked"] == "circuit_breaker"
