"""
Weather-related functions.
"""
import random
from typing import Dict, List


def get_weather(location: str) -> str:
    """
    Get weather information for a location.
    
    This is a realistic simulation that provides varied weather data
    based on the location and includes seasonal/geographic considerations.
    """
    # Normalize location
    location = location.strip().title()
    
    # Define weather patterns for different types of locations
    weather_patterns = {
        # Major cities with typical weather
        "new york": ["15°C, overcast with light rain", "22°C, partly cloudy", "8°C, clear and cold", "18°C, sunny with light winds"],
        "london": ["12°C, rainy and foggy", "16°C, cloudy with drizzle", "9°C, overcast", "14°C, partly sunny"],
        "tokyo": ["20°C, humid and cloudy", "25°C, sunny", "18°C, light rain", "23°C, partly cloudy"],
        "paris": ["17°C, partly cloudy", "13°C, light rain", "21°C, sunny", "11°C, overcast"],
        "sydney": ["24°C, sunny and warm", "28°C, clear skies", "22°C, partly cloudy", "26°C, sunny with ocean breeze"],
        "moscow": ["2°C, snowy", "-5°C, clear and very cold", "1°C, overcast", "-3°C, light snow"],
        "dubai": ["32°C, sunny and hot", "35°C, very hot and dry", "30°C, clear", "38°C, extremely hot"],
        "mumbai": ["28°C, humid and hot", "32°C, very humid", "26°C, monsoon rains", "30°C, hot and sticky"],
    }
    
    # Climate-based defaults for unknown locations
    climate_defaults = [
        "20°C, partly cloudy with light winds",
        "18°C, sunny with scattered clouds", 
        "22°C, overcast with chance of rain",
        "16°C, clear and mild",
        "24°C, warm and sunny",
        "14°C, cool with light breeze",
        "19°C, partly sunny",
        "21°C, mild and pleasant"
    ]
    
    # Check if we have specific data for this location
    location_key = location.lower()
    
    if location_key in weather_patterns:
        weather_condition = random.choice(weather_patterns[location_key])
    else:
        # For unknown locations, use climate defaults
        weather_condition = random.choice(climate_defaults)
    
    # Add some realistic details
    extras = [
        "Humidity: 65%",
        "Wind: 8 km/h from the west", 
        "Visibility: 10 km",
        "UV Index: Moderate",
        "Feels like 2°C warmer",
        "Barometric pressure: 1013 hPa"
    ]
    
    extra_detail = random.choice(extras)
    
    return f"Weather in {location}: {weather_condition}. {extra_detail}"


# Function metadata
WEATHER_FUNCTION = {
    "name": "get_weather",
    "description": "Get current weather information for a specific location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get weather for"
            }
        },
        "required": ["location"]
    },
    "function": get_weather
} 