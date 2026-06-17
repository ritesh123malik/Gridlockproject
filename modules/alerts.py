"""
modules/alerts.py
-----------------
Citizen Alert Generator. Crafts public-facing traffic advisories, expected delays,
detour routes, and travel alternative alerts for citizens during major events.
"""

from datetime import datetime

class CitizenAlertGenerator:
    def __init__(self):
        pass

    def generate_citizen_alert(self, event_type: str, location: str, expected_delay_min: float, 
                               detour_nodes: list[str], weather: str = "Sunny") -> dict:
        """
        Creates advisory texts for social media, maps API alerts, and SMS broadcasts.
        """
        now = datetime.now()
        start_hour = now.strftime("%I:%M %p")
        # Estimate event completion in 3 hours
        end_hour = datetime.fromtimestamp(now.timestamp() + 10800).strftime("%I:%M %p")
        
        title = f"🚦 TRAFFIC ADVISORY: {event_type.upper()} AT {location.upper()}"
        
        delay_msg = f"Expect severe delays of up to **{int(expected_delay_min)} minutes** on **{location}** and adjacent connectors."
        
        weather_alert = ""
        if weather in ["Light Rain", "Heavy Rain"]:
            weather_alert = "🌧️ **Rain Warning**: Road surfaces are wet with localized puddling. Reduced visibility. Drive with extreme caution."
        elif weather == "Fog":
            weather_alert = "🌫️ **Visibility Alert**: Fog conditions. Reduce speed and utilize fog lamps."

        detour_msg = "No alternative routes computed."
        if detour_nodes and len(detour_nodes) > 1:
            detour_msg = f"🗺️ **Detour Active**: Motorists are advised to bypass the gridlock area. Use **Route A Detour**: `{' → '.join(detour_nodes)}`."

        transit_msg = "🚇 **Transit Advisory**: Namma Metro and public bus systems are operating with increased frequency. We highly recommend leaving private cars/cabs behind and opting for transit to bypass CBD/corridor blockages."

        advisory_sms = f"TLENS ALERT: {event_type} at {location} starting {start_hour}. Delays ~{int(expected_delay_min)}m. Bypass via {detour_nodes[1] if len(detour_nodes) > 1 else 'adjoining streets'}. Prefer Metro."

        return {
            "title": title,
            "time_window": f"{start_hour} - {end_hour}",
            "delay_warning": delay_msg,
            "weather_warning": weather_alert,
            "detour_path": detour_msg,
            "transit_recommendation": transit_msg,
            "sms_advisory": advisory_sms
        }
