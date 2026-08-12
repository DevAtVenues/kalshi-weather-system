"""Flask app — serves the dashboard UI and the /api/data JSON endpoint."""
from __future__ import annotations

import re
import threading

from flask import Flask, jsonify, make_response, redirect, render_template

from .store import DataStore

# Cache: market ticker → resolved Kalshi URL (avoids repeated API calls)
_kalshi_url_cache: dict[str, str] = {}
_kalshi_url_lock  = threading.Lock()

app   = Flask(__name__)
_store: DataStore | None = None


@app.route("/")
def index():
    # No-store so a relaunch always loads the current JS. Prevents a stale cached
    # page from polling routes that no longer exist (the /api/picks 404 ghost).
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.route("/api/data")
def api_data():
    return jsonify(_store.snapshot())


@app.route("/api/refresh")
def api_refresh():
    import traceback, io, sys
    try:
        _store._refresh_forecasts()
    except Exception as exc:
        return jsonify({"status": "error", "phase": "forecasts", "error": str(exc), "traceback": traceback.format_exc()}), 500
    buf = io.StringIO()
    try:
        old_stdout = sys.stdout
        sys.stdout = buf
        _store._refresh_model_picks()
        sys.stdout = old_stdout
    except Exception as exc:
        sys.stdout = old_stdout
        return jsonify({"status": "error", "phase": "model_picks", "error": str(exc),
                        "output": buf.getvalue(), "traceback": traceback.format_exc()}), 500
    return jsonify({"status": "ok",
                    "last_forecast": _store.last_forecast_utc.isoformat() if _store.last_forecast_utc else None,
                    "last_model": _store.last_model_utc.isoformat() if _store.last_model_utc else None,
                    "model_output": buf.getvalue()[:2000]})


@app.route("/api/trades")
def api_trades():
    from .trades import load_trades, auto_grade_from_market, compute_stats
    from kalshi_weather.ingest.kalshi import _get
    import traceback
    try:
        trades = load_trades()
        # Auto-grade open trades from current market prices
        open_tickers = [t["ticker"] for t in trades if t.get("outcome") is None]
        snap: dict = {}
        for ticker in open_tickers:
            try:
                r = _get(f"/markets/{ticker}", {})
                m = r.get("market", {})
                snap[ticker] = {
                    "yes_bid": float(m.get("yes_bid_dollars") or 0.5),
                    "yes_ask": float(m.get("yes_ask_dollars") or 0.5),
                }
            except Exception:
                pass
        trades = auto_grade_from_market(trades, snap)
        return jsonify(compute_stats(trades))
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/api/kalshi-redirect/<path:ticker>")
def kalshi_redirect(ticker: str):
    """
    Resolve the correct Kalshi web URL for a market ticker and redirect.
    Kalshi's actual URL format:
      /markets/{series_lower}/{title-slug}/{event_lower}?op_market_ticker={TICKER_UPPER}
    """
    from kalshi_weather.ingest.kalshi import _get

    ticker = ticker.upper()
    with _kalshi_url_lock:
        if ticker in _kalshi_url_cache:
            return redirect(_kalshi_url_cache[ticker])

    try:
        market     = _get(f"/markets/{ticker}", {}).get("market", {})
        event_tk   = market.get("event_ticker", "")
        event      = _get(f"/events/{event_tk}", {}).get("event", {})
        series_tk  = event.get("series_ticker", "").lower()
        title      = event.get("title", "")

        # Slugify: drop date clause (" on ..."), lowercase, alphanum+hyphens only
        slug_base  = title.split(" on ")[0].rstrip("?").strip() if title else ""
        slug       = re.sub(r"[^a-z0-9]+", "-", slug_base.lower()).strip("-")

        event_lower = event_tk.lower()
        url = (
            f"https://kalshi.com/markets/{series_tk}/{slug}/{event_lower}"
            f"?op_market_ticker={ticker}"
        )
    except Exception:
        url = "https://kalshi.com/markets"

    with _kalshi_url_lock:
        _kalshi_url_cache[ticker] = url

    return redirect(url)


@app.route("/api/trades/add", methods=["POST"])
def api_add_trade():
    from .trades import add_trade
    import traceback
    try:
        body = __import__("flask").request.get_json(force=True)
        trade = add_trade(body)
        return jsonify({"ok": True, "id": trade["id"]})
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/api/trades/remove", methods=["POST"])
def api_remove_trade():
    from .trades import load_trades, save_trades
    import traceback
    try:
        body = __import__("flask").request.get_json(force=True)
        trades = load_trades()
        trades = [t for t in trades if t.get("id") != body.get("id")]
        save_trades(trades)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.route("/api/trades/grade", methods=["POST"])
def api_grade_trade():
    from .trades import grade_trade
    import traceback
    try:
        body = __import__("flask").request.get_json(force=True)
        ok = grade_trade(
            body["id"],
            actual_high=body.get("actual_high"),
            outcome=body.get("outcome"),
        )
        return jsonify({"ok": ok})
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


def create_app(stations_cfg: dict, poll_interval: int = 600) -> Flask:
    global _store
    _store = DataStore(stations_cfg, poll_interval)
    _store.start()
    return app
