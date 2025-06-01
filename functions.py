from datetime import datetime
from zoneinfo import ZoneInfo

# ── 1. Server-side Python tools which model can call ─────────────────────
def get_current_time() -> str:
    """Get current time in US Eastern Time."""
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def get_weather(location: str) -> str:
    return f"Weather in {location}: 25 °C, partly cloudy and  warm."

FUNCTIONS = {
    "get_current_time": {"fn": get_current_time, "params": {}},
    "get_weather": {"fn": get_weather, "params": {"location": str}},
}