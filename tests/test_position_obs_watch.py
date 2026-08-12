"""Adverse-obs position alerts (built 2026-08-03 after KPHL 79->87 ran into a
held NO->87 with no phone alert: watchlist was empty and no obs check existed).

Covers: loss_floor_f conventions, obs_check staging/dedup, and the
trade->watchlist auto-arm parser in log_to_sheets.
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import position_watch as pw
import log_to_sheets as lts


def test_loss_floor_conventions():
    # between BUY_NO loses when the high ENTERS the bracket (inclusive floor)
    assert pw.loss_floor_f({"strike_type": "between", "direction": "BUY_NO",
                            "floor": 92, "cap": 93}) == 92
    # between BUY_YES loses when the high overshoots the cap
    assert pw.loss_floor_f({"strike_type": "between", "direction": "BUY_YES",
                            "floor": 92, "cap": 93}) == 94
    # greater(>T) BUY_NO loses at T+1 (strict greater)
    assert pw.loss_floor_f({"strike_type": "greater", "direction": "BUY_NO",
                            "threshold": 87}) == 88
    # less(<T) BUY_YES loses at T (strict less)
    assert pw.loss_floor_f({"strike_type": "less", "direction": "BUY_YES",
                            "threshold": 77}) == 77
    # rising obs cannot hurt these; no alert path
    assert pw.loss_floor_f({"strike_type": "less", "direction": "BUY_NO",
                            "threshold": 77}) is None
    assert pw.loss_floor_f({"strike_type": "greater", "direction": "BUY_YES",
                            "threshold": 87}) is None


def _entry(**kw):
    e = {"ticker": "KXHIGHPHIL-26AUG03-T87", "direction": "BUY_NO",
         "strike_type": "greater", "threshold": 87, "station": "KPHL",
         "settlement_date": "2026-08-03", "lst_offset_h": -5}
    e.update(kw)
    return e


def _with_band(high_min, entry, st, today="2026-08-03"):
    with mock.patch.object(pw, "day_running_band",
                           return_value={"high_min_f": high_min, "high_max_f": high_min + 0.6,
                                         "latest_precise_f": high_min}):
        with mock.patch("position_watch.loss_floor_f", wraps=pw.loss_floor_f):
            import datetime as _dt
            class _FakeDT(_dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    return _dt.datetime(2026, 8, 3, 20, 0, tzinfo=_dt.timezone.utc)
            with mock.patch("datetime.datetime", _FakeDT):
                return pw.obs_check(entry, st)


def test_obs_stages_escalate_and_dedup():
    st = {}
    assert _with_band(84.0, _entry(), st) is None            # quiet below floor-2
    a = _with_band(86.0, _entry(), st)
    assert a and "WARN" in a                                  # 88-2
    assert _with_band(86.4, _entry(), st) is None             # same stage: no repeat
    a = _with_band(87.0, _entry(), st)
    assert a and "CRITICAL" in a                              # 88-1
    a = _with_band(88.0, _entry(), st)
    assert a and "BREACHED" in a
    assert _with_band(88.5, _entry(), st) is None             # terminal stage: silent


def test_trade_arms_watchlist_entry():
    t = {"date": "08/04/2026", "status": "Open", "position": "No",
         "market": "Miami High Temp, Aug 4 — Between 92-93°F (B92.5)",
         "entry_price": "$0.66"}
    e = lts.watchlist_entry_from_trade(t)
    assert e is not None
    assert e["ticker"] == "KXHIGHMIA-26AUG04-B92.5"
    assert e["direction"] == "BUY_NO"
    assert e["strike_type"] == "between" and e["floor"] == 92 and e["cap"] == 93
    assert e["station"] == "KMIA"


def test_settled_trade_not_armed():
    t = {"date": "08/03/2026", "status": "Settled", "position": "No",
         "market": "Miami High Temp — Between 92-93°F (B92.5)"}
    assert lts.watchlist_entry_from_trade(t) is None
