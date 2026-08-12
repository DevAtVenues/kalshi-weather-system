"""
Kalshi trading client — the ONLY module in the codebase that can place or
cancel orders. Everything else is read-only Bearer GETs; order endpoints
require RSA-PSS request signing, which lives here and nowhere else.

SAFETY MODEL (do not weaken):
  - Environment defaults to DEMO (demo-api.kalshi.co, play money).
  - Production requires BOTH KALSHI_EXEC_ENV=prod AND the explicit interlock
    KALSHI_EXEC_ALLOW_PROD=I_UNDERSTAND_THIS_TRADES_REAL_MONEY in .env.
    Enforced at construction, covered by the commit-gate smoke test.
  - Orders are always buy-side, limit, post-only (maker). Exits are
    hold-to-settlement — the timing study showed stops destroy the edge.

Env (.env — scheduled jobs inherit nothing else):
  KALSHI_EXEC_ENV                demo (default) | prod
  KALSHI_DEMO_KEY_ID             API key id from demo.kalshi.co settings
  KALSHI_DEMO_PRIVATE_KEY_PATH   path to the RSA private key PEM for that key
  KALSHI_PROD_KEY_ID / KALSHI_PROD_PRIVATE_KEY_PATH   (prod only)
"""
from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

_BASES = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
}
_API_PREFIX = "/trade-api/v2"          # signed path includes this prefix
_PROD_INTERLOCK = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"
_MIN_REQUEST_GAP_S = 0.35              # WAF lesson (2026-07-23): never burst


class ExchangeError(RuntimeError):
    """Raised on config problems or non-retryable API errors."""

    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(msg)
        self.status = status
        self.body = body


class KalshiExchange:
    """Signed Kalshi portfolio/trading API. One instance per run."""

    def __init__(self, env: str | None = None) -> None:
        self.env = (env or os.getenv("KALSHI_EXEC_ENV") or "demo").strip().lower()
        if self.env not in _BASES:
            raise ExchangeError(f"unknown KALSHI_EXEC_ENV {self.env!r} (demo|prod)")
        if self.env == "prod" and os.getenv("KALSHI_EXEC_ALLOW_PROD") != _PROD_INTERLOCK:
            raise ExchangeError(
                "refusing PROD execution: set KALSHI_EXEC_ALLOW_PROD="
                f"{_PROD_INTERLOCK} explicitly (demo needs nothing)")
        self.base = _BASES[self.env]

        pfx = "KALSHI_DEMO" if self.env == "demo" else "KALSHI_PROD"
        self.key_id = os.getenv(f"{pfx}_KEY_ID") or ""
        key_path = os.getenv(f"{pfx}_PRIVATE_KEY_PATH") or ""
        if not self.key_id or not key_path:
            raise ExchangeError(
                f"{pfx}_KEY_ID / {pfx}_PRIVATE_KEY_PATH unset — create an API key "
                f"at {'demo.kalshi.co' if self.env == 'demo' else 'kalshi.com'} "
                "settings and put both in .env")
        pem = Path(key_path).expanduser()
        if not pem.is_file():
            raise ExchangeError(f"private key file not found: {pem}")
        self._key = serialization.load_pem_private_key(pem.read_bytes(), password=None)
        self._last_request = 0.0

    # ── signing / transport ──────────────────────────────────────────────────

    def _sign(self, ts_ms: str, method: str, path: str) -> str:
        msg = f"{ts_ms}{method}{_API_PREFIX}{path}".encode()
        sig = self._key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY":       self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts_ms, method, path),
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "Content-Type":            "application/json",
        }

    def _request(self, method: str, path: str, params: dict | None = None,
                 body: dict | None = None) -> dict:
        """path excludes the /trade-api/v2 prefix and any query string."""
        gap = _MIN_REQUEST_GAP_S - (time.monotonic() - self._last_request)
        if gap > 0:
            time.sleep(gap)
        url = self.base + path
        for attempt in range(5):
            self._last_request = time.monotonic()
            resp = requests.request(method, url, params=params, json=body,
                                    headers=self._headers(method, path), timeout=15)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise ExchangeError(
                    f"{method} {path} -> {resp.status_code}: {resp.text[:300]}",
                    status=resp.status_code, body=resp.text[:1000])
            return resp.json() if resp.text else {}
        raise ExchangeError(f"{method} {path}: retries exhausted")

    # ── read endpoints ───────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self) -> list[dict]:
        return self._request("GET", "/portfolio/positions").get("market_positions", [])

    def get_resting_orders(self) -> list[dict]:
        return self._request("GET", "/portfolio/orders",
                             params={"status": "resting"}).get("orders", [])

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/portfolio/orders/{order_id}").get("order", {})

    def get_fills(self, min_ts: int | None = None) -> list[dict]:
        params = {"limit": 200}
        if min_ts:
            params["min_ts"] = int(min_ts)
        return self._request("GET", "/portfolio/fills", params=params).get("fills", [])

    def get_settlements(self, limit: int = 100) -> list[dict]:
        return self._request("GET", "/portfolio/settlements",
                             params={"limit": limit}).get("settlements", [])

    # ── order endpoints ──────────────────────────────────────────────────────

    def create_order(self, ticker: str, side: str, count: int, price_cents: int,
                     expiration_ts: int | None = None,
                     client_order_id: str | None = None) -> dict:
        """Buy-side post-only limit order. side is 'yes' or 'no'; price_cents is
        the limit price of THAT side (1-99). Returns the API's order object."""
        if side not in ("yes", "no"):
            raise ExchangeError(f"bad side {side!r}")
        if not (1 <= int(price_cents) <= 99):
            raise ExchangeError(f"bad price {price_cents} for {ticker} (must be 1-99c)")
        if int(count) < 1:
            raise ExchangeError(f"bad count {count} for {ticker}")
        body: dict = {
            "ticker":          ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side":            side,
            "action":          "buy",
            "count":           int(count),
            "type":            "limit",
            "post_only":       True,     # never cross: maker or rejected
            ("yes_price" if side == "yes" else "no_price"): int(price_cents),
        }
        if expiration_ts:
            body["expiration_ts"] = int(expiration_ts)   # server-side TTL cancel
        return self._request("POST", "/portfolio/orders", body=body).get("order", {})

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
