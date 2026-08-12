"""Health watchdog: alert cooldown state machine + the checks' pure edges."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import health_check as hc


def _prime(tmp_path, monkeypatch, reds, ts):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"reds": reds, "ts": ts}))
    monkeypatch.setattr(hc, "STATE", state)


def test_new_red_alerts(tmp_path, monkeypatch):
    _prime(tmp_path, monkeypatch, [], 0)
    assert hc.should_alert(["orderbook logger"], now=1000) == (True, "red")


def test_same_reds_within_cooldown_stay_quiet(tmp_path, monkeypatch):
    _prime(tmp_path, monkeypatch, ["orderbook logger"], 1000)
    assert hc.should_alert(["orderbook logger"], now=1000 + 3600) == (False, None)


def test_same_reds_realert_after_cooldown(tmp_path, monkeypatch):
    _prime(tmp_path, monkeypatch, ["orderbook logger"], 1000)
    assert hc.should_alert(["orderbook logger"],
                           now=1000 + hc.ALERT_COOLDOWN_S + 1) == (True, "red")


def test_changed_red_set_alerts_immediately(tmp_path, monkeypatch):
    _prime(tmp_path, monkeypatch, ["orderbook logger"], 1000)
    assert hc.should_alert(["orderbook logger", "kalshi canary"],
                           now=1001) == (True, "red")


def test_recovery_pushes_once(tmp_path, monkeypatch):
    _prime(tmp_path, monkeypatch, ["orderbook logger"], 1000)
    assert hc.should_alert([], now=1001) == (True, "recovered")
    _prime(tmp_path, monkeypatch, [], 1001)      # state after the recovery push
    assert hc.should_alert([], now=1002) == (False, None)


def test_missing_state_treated_as_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "STATE", tmp_path / "nope.json")
    assert hc.should_alert([], now=1000) == (False, None)
    assert hc.should_alert(["x"], now=1000) == (True, "red")


def test_crashing_check_reads_as_red(monkeypatch):
    monkeypatch.setattr(hc, "CHECKS", [("boom", lambda: 1 / 0)])
    results = hc.run_checks()
    assert results[0][0] == "RED" and "crashed" in results[0][2]


def _fake_pmset(monkeypatch, stdout):
    class R:
        pass
    r = R()
    r.stdout = stdout
    monkeypatch.setattr(hc.subprocess, "run", lambda *a, **k: r)


def test_power_on_ac_is_ok(monkeypatch):
    _fake_pmset(monkeypatch, "Now drawing from 'AC Power'\n -InternalBattery-0\t80%; charging;\n")
    status, detail = hc.check_power_source()
    assert status == "OK" and "80%" in detail


def test_power_on_battery_warns(monkeypatch):
    _fake_pmset(monkeypatch, "Now drawing from 'Battery Power'\n -InternalBattery-0\t95%; discharging;\n")
    status, detail = hc.check_power_source()
    assert status == "WARN" and "95%" in detail


def test_power_low_battery_is_red(monkeypatch):
    _fake_pmset(monkeypatch, "Now drawing from 'Battery Power'\n -InternalBattery-0\t12%; discharging;\n")
    status, _ = hc.check_power_source()
    assert status == "RED"


def test_orderbook_rollover_grace(tmp_path, monkeypatch):
    """Missing today-file is OK (not RED) iff yesterday's file is minutes-fresh."""
    from datetime import datetime, timedelta, timezone
    monkeypatch.setattr(hc, "ORDERBOOK", tmp_path)
    today = datetime.now(timezone.utc).date()
    prev = tmp_path / f"{today - timedelta(days=1)}.parquet"
    prev.write_bytes(b"x")                                    # mtime = now → fresh
    status, detail = hc.check_orderbook_logger()
    assert status == "OK" and "rollover" in detail


def test_orderbook_missing_today_and_stale_yesterday_is_red(tmp_path, monkeypatch):
    import os, time
    from datetime import datetime, timedelta, timezone
    monkeypatch.setattr(hc, "ORDERBOOK", tmp_path)
    today = datetime.now(timezone.utc).date()
    prev = tmp_path / f"{today - timedelta(days=1)}.parquet"
    prev.write_bytes(b"x")
    old = time.time() - 3600                                  # stale: 60 min old
    os.utime(prev, (old, old))
    status, _ = hc.check_orderbook_logger()
    assert status == "RED"
