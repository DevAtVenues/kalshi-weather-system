"""
Push notifications via ntfy.sh.

POLICY (enforced upstream in DataStore._check_transitions): a push fires ONLY when
a contract newly LOCKS and still has actionable edge — the "hop on it now" case.
Near-locks, model edges and heuristics are reviewed in the dashboard, not pushed.
This module is just the delivery mechanism; it stays generic so the policy lives
in one place.

Set NTFY_TOPIC in .env to enable.  The dashboard will POST to:
    https://ntfy.sh/{NTFY_TOPIC}

Install the ntfy app on your phone and subscribe to the same topic.
No account or payment required for basic use.
"""
from __future__ import annotations

import json
import os
import logging
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Risk-halt check: this is the single choke point every trade push flows through
# (run_live cron, run_live_engine, check_locks, dashboard), so the daily circuit
# breaker is enforced HERE. Read the state file directly — cheap, cross-process,
# and avoids importing the risk module into the delivery layer.
_RISK_STATE = Path(__file__).parents[3] / "data" / "risk_state.json"


def _risk_halted_today() -> bool:
    try:
        s = json.loads(_RISK_STATE.read_text())
        return bool(s.get("halted")) and s.get("date") == date.today().isoformat()
    except Exception:
        return False

_NTFY_BASE = "https://ntfy.sh"

# Category priority for transition detection (higher = more urgent)
_PRIORITY = {"gfs_edge": 1, "watchlist": 0, "near_lock": 2, "lock": 3, "model_edge": 1}

# ntfy priority and tag per category
_NTFY_CFG = {
    "lock": {
        "priority": "urgent",
        "tags":     "rotating_light,money_with_wings",
        "title":    "LOCK",
    },
    "near_lock": {
        "priority": "high",
        "tags":     "bell,chart_with_upwards_trend",
        "title":    "Near-Lock",
    },
    "model_edge": {
        "priority": "high",
        "tags":     "chart_with_upwards_trend,moneybag",
        "title":    "Model Edge",
    },
}


def _topic() -> str | None:
    return os.getenv("NTFY_TOPIC") or None


def notify_transition(pick: dict, old_cat: str | None) -> bool:
    """
    Fire a push notification when a pick reaches near_lock or lock status.

    Returns False ONLY when the push was suppressed by the risk circuit breaker —
    callers use that to skip marking the pick as notified, so counters and the
    notified_today dedup state stay honest under a halt. Every other outcome
    (delivered, no NTFY_TOPIC configured, delivery error) returns True so the
    existing dedup behavior for topic-less setups is unchanged.
    """
    cat = pick.get("cat", "")
    cfg = _NTFY_CFG.get(cat)
    if cfg is None:
        return True  # gfs_edge / watchlist don't warrant a push

    if _risk_halted_today():
        log.warning("risk circuit breaker tripped — suppressing %s push for %s",
                    cat, pick.get("contract_label", "?"))
        return False

    topic = _topic()
    if not topic:
        return True

    city    = pick.get("city", "?")
    label   = pick.get("contract_label", pick.get("ticker", "?"))
    action  = pick.get("action", "?")
    reason  = pick.get("reason", "")
    bid     = pick.get("yes_bid")
    rhi     = pick.get("day_h") or pick.get("temp_f")

    bid_str = f"  bid {bid}¢" if bid is not None else ""
    rhi_str = f"  rhi {rhi:.1f}°F" if rhi is not None else ""
    transition = f" (was {old_cat})" if old_cat else ""

    message = f"{action} {label}{bid_str}{rhi_str}\n{reason}{transition}"

    try:
        requests.post(
            f"{_NTFY_BASE}/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title":    f"{cfg['title']}: {city}",
                "Priority": cfg["priority"],
                "Tags":     cfg["tags"],
            },
            timeout=5,
        )
    except Exception as exc:
        log.warning("ntfy notification failed: %s", exc)
    return True


def _ascii_title(title: str) -> str:
    """HTTP headers are latin-1; an emoji in Title makes requests RAISE and the
    push silently dies (bit the feed-stale alert). Emoji belong in Tags/body."""
    return title.encode("ascii", "ignore").decode().strip()


def notify_health(title: str, body: str, priority: str = "high",
                  tags: str = "rotating_light") -> None:
    """System-health watchdog push (scripts/health_check.py). Deliberately does NOT
    consult the risk circuit breaker: a halt suppresses trade pushes, but a broken
    logger/feed must always reach the phone."""
    topic = _topic()
    if not topic:
        return
    try:
        requests.post(
            f"{_NTFY_BASE}/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": _ascii_title(title), "Priority": priority, "Tags": tags},
            timeout=5,
        )
    except Exception as exc:
        log.warning("ntfy health alert failed: %s", exc)


def notify_feed_stale(stale_min: float) -> None:
    """
    Watchdog alert: the observation feed has stopped updating, so lock detection
    (and therefore the lock pushes) may be silently broken. Tell the user to restart.
    Fired once per stale episode by the store's watchdog thread.
    """
    topic = _topic()
    if not topic:
        return
    msg = (f"Observation feed hasn't updated in {stale_min:.0f} min — lock detection "
           f"may be stalled and you could miss a lock. Restart the dashboard.")
    try:
        requests.post(
            f"{_NTFY_BASE}/{topic}",
            data=msg.encode("utf-8"),
            headers={"Title": "Weather feed stale", "Priority": "high", "Tags": "warning"},
            timeout=5,
        )
    except Exception as exc:
        log.warning("ntfy stale-feed alert failed: %s", exc)
