"""One-off: update Kalshi Market column for trades 6–14 to abbreviated format."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from kalshi_weather.sheets_logger import _get_worksheet

NAMES = {
    6:  "Highest Temp NYC, 6/14: 85°–86°",
    7:  "Highest Temp CHI, 6/14: 73°–74°",
    8:  "Lowest Temp SEA, 6/15: ≥64°",
    9:  "Highest Temp MIA, 6/15: 91°–92°",
    10: "Highest Temp MIA, 6/16: 94°–95°",
    11: "Highest Temp PHL, 6/16: 77°–78°",
    12: "Highest Temp NYC, 6/16: 75°–76°",
    13: "Highest Temp MIA, 6/17: 93°–94°",
    14: "Highest Temp AUS, 6/16: 88°–89°",
}

ws = _get_worksheet()
col_a = ws.col_values(1)
for i, val in enumerate(col_a, 1):
    stripped = val.strip()
    if stripped.isdigit() and int(stripped) in NAMES:
        trade_num = int(stripped)
        ws.update_cell(i, 3, NAMES[trade_num])
        print(f"  trade #{trade_num} → {NAMES[trade_num]}")
print("Done.")
