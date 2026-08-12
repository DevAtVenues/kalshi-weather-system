"""
Google Sheets trade logger.

Appends trade rows to the next empty row in the trade log spreadsheet.

Auth: gspread OAuth2 — first run opens a browser, token cached at
      ~/.config/gspread/authorized_user.json for all future runs.
"""
from __future__ import annotations

from typing import Any

import gspread

SPREADSHEET_ID  = "<your-google-sheet-id>"   # set per deployment
WORKSHEET_GID   = 1863038233

# Column order must match the sheet header exactly
COLUMNS = [
    "Trade #",
    "Date",
    "Kalshi Market",
    "Position (Yes/No)",
    "Amount Committed ($)",
    "Entry Price ($)",
    "Exit/Settlement Price ($)",
    "Fees ($)",
    "Net P/L ($)",
    "Status",
    "Screenshot Link",
    "Notes",
]


def _get_worksheet() -> gspread.Worksheet:
    gc = gspread.oauth()
    ss = gc.open_by_key(SPREADSHEET_ID)
    return ss.get_worksheet_by_id(WORKSHEET_GID)


def _last_trade_row(ws: gspread.Worksheet) -> int:
    """Return the 1-based row index of the last row containing a numeric trade #."""
    col_a = ws.col_values(1)
    last_row = 1
    for i, val in enumerate(col_a, 1):
        if val.strip().isdigit():
            last_row = i
    return last_row


def _next_trade_number(ws: gspread.Worksheet) -> int:
    col_a = ws.col_values(1)
    last = 0
    for val in col_a[1:]:  # skip header
        if val.strip().isdigit():
            last = max(last, int(val.strip()))
    return last + 1


def log_trades(trades: list[dict[str, Any]]) -> None:
    """
    Write one or more trade rows directly below the last numbered trade row.

    Each trade dict may contain:
        date              str  e.g. "06/15/2026"
        market            str  abbreviated, e.g. "Highest Temp NYC, 6/16: 75°–76°"
        position          str  "Yes" or "No"
        amount_committed  str  e.g. "$3.71"
        entry_price       str  e.g. "$0.53"
        settlement_price  str  e.g. "$1.00" or "-"
        fees              str  e.g. "$0.08" or "-"
        net_pl            str  e.g. "$3.29" or "($4.49)"
        status            str  "Settled" | "Open" | "Closed"
        screenshot        str  URL or ""
        notes             str  free text
    """
    ws = _get_worksheet()
    next_row = _last_trade_row(ws) + 1
    trade_num = _next_trade_number(ws)

    for trade in trades:
        row = [
            trade_num,
            trade.get("date", ""),
            trade.get("market", ""),
            trade.get("position", ""),
            trade.get("amount_committed", ""),
            trade.get("entry_price", ""),
            trade.get("settlement_price", "-"),
            trade.get("fees", "-"),
            trade.get("net_pl", ""),
            trade.get("status", "Open"),
            trade.get("screenshot", ""),
            trade.get("notes", ""),
        ]
        ws.update(f"A{next_row}:L{next_row}", [row], value_input_option="USER_ENTERED")
        print(f"  logged trade #{trade_num}: {trade.get('market','')[:60]}")
        next_row += 1
        trade_num += 1
