"""
Time-related functions.
"""
from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_time() -> str:
    """Get current time in US Eastern Time."""
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


# Function metadata
TIME_FUNCTION = {
    "name": "get_current_time",
    "description": "Get the current time in US Eastern Time",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    },
    "function": get_current_time
} 