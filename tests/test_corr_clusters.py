"""Cluster correlation cap: same-day same-SYSTEM cities are one weather bet.
Three Texas cities at the per-city cap must not stack 3x exposure to one
ridge — the cluster cap binds across them. Purely protective: it can only
reduce size, never increase it."""
import pandas as pd

import kalshi_weather.live.risk as risk_mod
from kalshi_weather.live.risk import (
    CORR_CLUSTERS, RiskManager, _MARKET_CITY_NAMES, cluster_of)


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(risk_mod, "CONFIG_PATH", tmp_path / "risk_config.json")
    monkeypatch.setattr(risk_mod, "STATE_PATH", tmp_path / "risk_state.json")


def _sig(city, prob=0.10, mid=0.50, direction="BUY_NO", date="2026-07-21"):
    return pd.Series({"city": city, "settlement_date": date,
                      "prob_estimate": prob, "market_mid": mid,
                      "direction": direction})


def test_every_tradeable_city_maps_to_exactly_one_cluster():
    seen: dict[str, str] = {}
    for name, cities in CORR_CLUSTERS.items():
        for c in cities:
            assert c not in seen, f"{c} in both {seen[c]} and {name}"
            seen[c] = name
    for code in set(_MARKET_CITY_NAMES.values()):
        assert code in seen, f"{code} has no cluster"
    assert cluster_of("XYZ") == "city:XYZ"      # unknown → singleton


def test_cluster_cap_binds_across_same_system_cities(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    rm = RiskManager()      # defaults: $250 bankroll → city cap $12.50, cluster cap $12.50
    # Dallas already holds the full cluster allowance for the date…
    rm.record_fill("DAL", "2026-07-21", 25, 12.50)
    # …so Austin (same southern_plains cluster, same date) is BLOCKED even
    # though Austin's own per-city exposure is zero.
    out = rm.size_trade(_sig("AUS"))
    assert out["blocked"] == "corr_cluster_full" and out["contracts"] == 0
    # A different cluster (Chicago) on the same date is unaffected.
    out = rm.size_trade(_sig("CHI"))
    assert out["blocked"] is None and out["contracts"] > 0
    # And the same cluster on a DIFFERENT date is unaffected.
    out = rm.size_trade(_sig("SAT", date="2026-07-22"))
    assert out["blocked"] is None and out["contracts"] > 0


def test_cluster_cap_trims_partial_headroom(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    rm = RiskManager()
    # DAL holds $9 of the $12.50 cluster allowance → SAT gets at most $3.50,
    # which is tighter than both its per-trade cap and its own city cap.
    rm.record_fill("DAL", "2026-07-21", 18, 9.00)
    out = rm.size_trade(_sig("SAT", prob=0.05, mid=0.60))   # big edge wants > $3.50
    assert out["cap"] == "corr_cluster"
    assert out["dollars"] <= 3.50 + 1e-9


def test_same_city_exposure_counts_once_not_twice(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    rm = RiskManager()
    # City exposure IS cluster exposure for a singleton cluster — no
    # double-count blocking: SEA with $6 used still sizes (city remaining
    # $6.50, cluster remaining $6.50).
    rm.record_fill("SEA", "2026-07-21", 12, 6.00)
    out = rm.size_trade(_sig("SEA", prob=0.05, mid=0.60))
    assert out["blocked"] is None
    assert out["dollars"] <= 6.50 + 1e-9
