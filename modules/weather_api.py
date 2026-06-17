"""
modules/weather_api.py
-----------------------
Live weather integration using OpenWeatherMap free API.
Falls back to a deterministic simulated value if no API key is provided.
"""

import random
from datetime import datetime

# Mapping OWM condition codes → TrafficLens weather strings
OWM_CONDITION_MAP = {
    range(200, 300): "Heavy Rain",   # Thunderstorm
    range(300, 400): "Light Rain",   # Drizzle
    range(500, 502): "Light Rain",   # Light rain
    range(502, 600): "Heavy Rain",   # Heavy rain
    range(600, 700): "Fog",          # Snow (rare for Bengaluru, treat as Fog)
    range(700, 800): "Fog",          # Atmosphere (mist, fog, haze)
    range(800, 801): "Sunny",        # Clear sky
    range(801, 900): "Sunny",        # Partly cloudy → treat as Sunny
}

# Seasonal probabilities for Bengaluru simulation (month → [Sunny, Light Rain, Heavy Rain, Fog])
_SEASONAL = {
    1: [0.6, 0.1, 0.0, 0.3],  # Jan — winter morning fog
    2: [0.7, 0.1, 0.0, 0.2],
    3: [0.8, 0.1, 0.0, 0.1],
    4: [0.6, 0.2, 0.2, 0.0],  # Apr — pre-monsoon
    5: [0.5, 0.3, 0.2, 0.0],
    6: [0.3, 0.3, 0.4, 0.0],  # Jun-Sep — monsoon
    7: [0.2, 0.3, 0.5, 0.0],
    8: [0.2, 0.3, 0.5, 0.0],
    9: [0.3, 0.3, 0.4, 0.0],
    10: [0.4, 0.3, 0.3, 0.0],
    11: [0.6, 0.2, 0.1, 0.1],
    12: [0.6, 0.1, 0.0, 0.3],
}

_CONDITIONS = ["Sunny", "Light Rain", "Heavy Rain", "Fog"]


class WeatherFetcher:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key.strip()

    def get_weather(self, lat: float = 12.9716, lon: float = 77.5946) -> dict:
        """
        Returns weather dict: {condition, description, temperature_c, humidity_pct}.
        Uses OWM API if key is provided, else returns seasonally-weighted simulation.
        """
        if self.api_key:
            return self._fetch_live(lat, lon)
        return self._simulate()

    def _fetch_live(self, lat: float, lon: float) -> dict:
        try:
            import urllib.request
            import json
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            )
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            code = data["weather"][0]["id"]
            description = data["weather"][0]["description"].title()
            temp = round(data["main"]["temp"], 1)
            humidity = data["main"]["humidity"]
            condition = self._map_code(code)
            return {
                "condition": condition,
                "description": description,
                "temperature_c": temp,
                "humidity_pct": humidity,
                "source": "OpenWeatherMap Live"
            }
        except Exception as e:
            # Graceful fallback
            result = self._simulate()
            result["source"] = f"Simulated (API error: {e})"
            return result

    def _simulate(self) -> dict:
        month = datetime.now().month
        probs = _SEASONAL.get(month, [0.5, 0.2, 0.2, 0.1])
        condition = random.choices(_CONDITIONS, weights=probs, k=1)[0]
        # Plausible Bengaluru temperatures by month
        temp_range = {1: (15, 22), 2: (17, 25), 3: (20, 28), 4: (22, 32),
                      5: (22, 31), 6: (20, 28), 7: (19, 26), 8: (19, 26),
                      9: (19, 27), 10: (19, 27), 11: (17, 25), 12: (15, 23)}
        lo, hi = temp_range.get(datetime.now().month, (18, 27))
        temp = round(random.uniform(lo, hi), 1)
        return {
            "condition": condition,
            "description": f"{condition} (Bengaluru Simulation)",
            "temperature_c": temp,
            "humidity_pct": random.randint(45, 90),
            "source": "Seasonally-Weighted Simulation"
        }

    def _map_code(self, code: int) -> str:
        for r, label in OWM_CONDITION_MAP.items():
            if code in r:
                return label
        return "Sunny"
