"""
modules/chat_assistant.py
-------------------------
TrafficGPT – simple rules-based NLP chat assistant.
Interprets traffic questions from users and retrieves active planning and simulation metrics.
"""

class TrafficChatAssistant:
    def __init__(self, forecast: dict, detours: dict, command_order: dict, sentiment: dict = None):
        self.forecast = forecast
        self.detours = detours
        self.command = command_order
        self.sentiment = sentiment or {"mood": "Neutral", "sentiment_score": 0.50, "estimated_tweets": 0}

    def respond(self, query: str) -> str:
        """
        Processes query strings and routes them to rule-matching intents.
        """
        q = query.lower().strip()
        
        if not q:
            return "I am TrafficGPT. Ask me about delays, route detours, officer deployments, or public sentiment!"

        # Delay & Travel Time Intents
        if any(w in q for w in ["delay", "wait", "travel time", "minutes", "duration"]):
            return (
                f"📊 **Expected Delay Forecast:** The predicted delay at {self.command.get('location_corridor')} "
                f"is currently **{self.forecast.get('expected_delay')} minutes** (estimated with {self.forecast.get('confidence')}% ML model confidence)."
            )

        # Route Detour Intents
        if any(w in q for w in ["route", "detour", "bypass", "direction", "alternative"]):
            nodes = self.detours.get("route_a", {}).get("nodes", [])
            dist = self.detours.get("route_a", {}).get("distance_km", 0.0)
            time_est = self.detours.get("route_a", {}).get("travel_time_min", 0.0)
            
            if nodes:
                nodes_str = " ➔ ".join(nodes)
                return (
                    f"🛣️ **Bypass Detour Directive:** To avoid the bottleneck on {self.command.get('location_corridor')}, "
                    f"we recommend taking **Route A**: `{nodes_str}` ({dist} km | ~{time_est} min transit)."
                )
            else:
                return "⚠️ **Bypass Detour Directive:** No alternative detour route was found in the corridor connectivity network."

        # Officer & Patrol Intents
        if any(w in q for w in ["officer", "police", "deployment", "manpower", "patrol"]):
            total = self.command.get("officers_total", 0)
            directives = self.command.get("directives", {}).get("patrols", [])
            dir_text = "\n".join([f"- {d}" for d in directives[:3]])
            
            return (
                f"👮 **Precinct Dispatch Summary:** A total of **{total} officers** are deployed. "
                f"Directives include:\n{dir_text}\n(Check the dispatch logs for detailed station-level borrowing plans)."
            )

        # Congestion Risk & Gridlock Intents
        if any(w in q for w in ["risk", "congestion", "gridlock", "saturated"]):
            risk = self.forecast.get("congestion_risk", 0.0)
            return (
                f"🚨 **Gridlock Severity Alert:** The current congestion risk level is **{risk}%**. "
                f"The incident epicenter corridor is flagged as: **{'CRITICAL' if risk > 75 else ('CONGESTED' if risk > 45 else 'STABLE')}**."
            )

        # Weather & Environment Intents
        if any(w in q for w in ["weather", "rain", "fog", "sunny"]):
            w_state = self.forecast.get("weather", "Sunny")
            return (
                f"🌧️ **Environmental Conditions:** The simulation is running with **{w_state}** weather parameters. "
                f"Poor weather compounds bottleneck densities and increases fuel waste factor."
            )

        # Public Sentiment Intents
        if any(w in q for w in ["sentiment", "public", "tweets", "mood", "opinion"]):
            return (
                f"💬 **Social Sentiment Score:** Public response is **{self.sentiment['mood']}** "
                f"({int(self.sentiment['sentiment_score'] * 100)}% positive rating) across an estimated "
                f"**{self.sentiment['estimated_tweets']} social posts**."
            )

        # General help fallback
        return (
            "🤖 **TrafficGPT Help Command:** I am a decision-support NLP agent. You can ask queries like:\n"
            "- *What is the expected delay right now?*\n"
            "- *Give me the detour routing directions.*\n"
            "- *How many officers are deployed at the stations?*\n"
            "- *What is the congestion risk and public sentiment?*"
        )
