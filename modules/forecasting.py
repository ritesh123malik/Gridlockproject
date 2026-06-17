"""
modules/forecasting.py
----------------------
Extended Event forecasting system. Predicts expected delay, congestion risk,
and required resources, incorporating weather, lifecycle stages, transport mode
share estimates, economic loss metrics, and an Explainable AI (XAI) breakdown.
"""

import math
import pandas as pd
from pathlib import Path
from modules.historical import load_corridor_centroids, load_corridor_scores
from modules.ml_model import HybridMLPredictor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EVENT_MULTIPLIERS = {
    "Political Rally": 1.65,
    "Festival": 1.35,
    "Sporting Event": 1.25,
    "Concert": 1.20,
    "VIP Movement": 1.80,
}

WEATHER_MULTIPLIERS = {
    "Sunny": 1.0,
    "Light Rain": 1.25,
    "Heavy Rain": 1.55,
    "Fog": 1.35
}

LIFECYCLE_MULTIPLIERS = {
    "Pre-event": 0.80,
    "During-event": 1.0,
    "Exit-wave": 1.45,
    "Post-event": 0.50
}

# Estimated transport split by event type
MODE_SHARES = {
    "Political Rally": {"Car/Cab": 0.15, "Public Transit": 0.50, "Two-Wheeler": 0.25, "Walk/Other": 0.10},
    "Festival": {"Car/Cab": 0.25, "Public Transit": 0.35, "Two-Wheeler": 0.30, "Walk/Other": 0.10},
    "Sporting Event": {"Car/Cab": 0.35, "Public Transit": 0.40, "Two-Wheeler": 0.20, "Walk/Other": 0.05},
    "Concert": {"Car/Cab": 0.40, "Public Transit": 0.30, "Two-Wheeler": 0.25, "Walk/Other": 0.05},
    "VIP Movement": {"Car/Cab": 0.70, "Public Transit": 0.10, "Two-Wheeler": 0.15, "Walk/Other": 0.05}
}

# Average passengers per vehicle class
OCCUPANCIES = {
    "Car/Cab": 2.0,
    "Public Transit": 40.0, # Bus/Metro
    "Two-Wheeler": 1.2,
}

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

class EventImpactForecaster:
    def __init__(self):
        self.centroids = load_corridor_centroids()
        self.scores_df = load_corridor_scores()
        self.scores_lookup = self.scores_df.set_index("corridor")["impact_score"].to_dict()
        self.max_score = self.scores_df["impact_score"].max() if not self.scores_df.empty else 1.0
        self.ml_predictor = HybridMLPredictor()

    def predict_impact(self, event_type: str, crowd_size: int, location_corridor: str, 
                       weather: str = "Sunny", lifecycle: str = "During-event", 
                       correction_factor: float = 0.0) -> dict:
        """
        Calculates:
          - Mode Share Splits & Vehicles Generated
          - Hybrid Machine Learning Predictions (XGBoost Emulator)
          - Weather & Lifecycle Multipliers
          - Congestion Risk & Expected Delay
          - Economic Loss (Lakhs, Fuel, Time)
          - Affected Corridors Proximities
          - Explainable AI (XAI) Point Breakdown
          - Officer, Barricade & Signage counts
        """
        multiplier = EVENT_MULTIPLIERS.get(event_type, 1.0)
        weather_mult = WEATHER_MULTIPLIERS.get(weather, 1.0)
        lifecycle_mult = LIFECYCLE_MULTIPLIERS.get(lifecycle, 1.0)

        # 1. Mode Share Estimation
        shares = MODE_SHARES.get(event_type, {"Car/Cab": 0.3, "Public Transit": 0.4, "Two-Wheeler": 0.2, "Walk/Other": 0.1})
        vehicles_generated = int(
            (crowd_size * shares["Car/Cab"] / OCCUPANCIES["Car/Cab"]) +
            (crowd_size * shares["Public Transit"] / OCCUPANCIES["Public Transit"]) +
            (crowd_size * shares["Two-Wheeler"] / OCCUPANCIES["Two-Wheeler"])
        )

        # 2. Get location centroid
        target_row = self.centroids[self.centroids["corridor"] == location_corridor]
        if not target_row.empty:
            target_lat = target_row.iloc[0]["centroid_lat"]
            target_lon = target_row.iloc[0]["centroid_lon"]
        else:
            target_lat, target_lon = 12.9716, 77.5946

        # Look up corridor historical prior
        corridor_score = self.scores_lookup.get(location_corridor, 0.0)
        prior_multiplier = 0.5 + (corridor_score / self.max_score) if self.max_score > 0 else 1.0

        # 3. Machine Learning predictions (KNN Regression)
        ml_out = self.ml_predictor.predict(event_type, crowd_size, location_corridor, weather, lifecycle)
        predicted_risk = ml_out["risk"]
        base_delay = ml_out["delay"]
        confidence = ml_out["confidence"]

        # Apply correction factor to ML delay output
        predicted_delay = base_delay * (1.0 + correction_factor)
        predicted_delay = max(2.0, round(predicted_delay, 1))
        predicted_risk = max(10.0, min(99.0, round(predicted_risk, 1)))

        # 5. Economic Impact Calculator
        # Background traffic volume scale by prior corridor importance
        background_flow = int(3000 * prior_multiplier)
        affected_vehicles = int(vehicles_generated + background_flow)
        delay_hours = predicted_delay / 60.0
        
        # Idling wastes ~0.6L of fuel per hour. Rain increases fuel consumption rate.
        fuel_rate = 0.6 * (1.1 if weather != "Sunny" else 1.0)
        fuel_wasted_liters = round(affected_vehicles * fuel_rate * delay_hours, 1)
        
        # Values: Fuel = ₹102/L, Citizen Time value = ₹350/hour
        time_cost = delay_hours * affected_vehicles * 350.0
        fuel_cost = fuel_wasted_liters * 102.0
        economic_loss_inr = time_cost + fuel_cost
        economic_loss_lakhs = round(economic_loss_inr / 100000.0, 2)

        # 6. Affected Corridors within 5.5 km
        affected = []
        for _, row in self.centroids.iterrows():
            dist = _haversine_km(target_lat, target_lon, row["centroid_lat"], row["centroid_lon"])
            if dist <= 5.5:
                affected.append({
                    "corridor": row["corridor"],
                    "distance_km": round(dist, 2),
                    "impact_rank": int(self.scores_lookup.get(row["corridor"], 0) / 10) + 1
                })
        affected = sorted(affected, key=lambda x: x["distance_km"])

        # 7. Resources Needed
        officers = max(5, int((vehicles_generated / 150.0) * multiplier * prior_multiplier * weather_mult))
        barricades = max(2, int(officers * 0.4))
        signage = max(2, int(officers * 0.3))

        # 8. Explainable AI (XAI) Point Breakdown
        # Distributes the risk points logically so they sum up to predicted_risk
        raw_hist = 20.0 * prior_multiplier
        raw_crowd = min(40.0, (vehicles_generated / 60.0) * multiplier)
        raw_type = 10.0 * multiplier
        raw_weather = 15.0 * (weather_mult - 0.8)
        raw_lifecycle = 15.0 * (lifecycle_mult - 0.3)
        
        raw_sum = raw_hist + raw_crowd + raw_type + raw_weather + raw_lifecycle
        
        xai_breakdown = {
            "Historical Corridor Prior": round((raw_hist / raw_sum) * predicted_risk, 1),
            "Crowd & Mode Share Size": round((raw_crowd / raw_sum) * predicted_risk, 1),
            "Event Type Severity": round((raw_type / raw_sum) * predicted_risk, 1),
            "Weather Conditions": round((raw_weather / raw_sum) * predicted_risk, 1),
            "Lifecycle Phase": round((raw_lifecycle / raw_sum) * predicted_risk, 1)
        }

        return {
            "congestion_risk": predicted_risk,
            "expected_delay": predicted_delay,
            "confidence": confidence,
            "affected_corridors": affected,
            "mode_shares": {k: round(v * 100, 1) for k, v in shares.items()},
            "vehicles_generated": vehicles_generated,
            "weather": weather,
            "lifecycle": lifecycle,
            "economics": {
                "affected_vehicles": affected_vehicles,
                "fuel_wasted_liters": fuel_wasted_liters,
                "economic_loss_lakhs": economic_loss_lakhs
            },
            "xai_breakdown": xai_breakdown,
            "resource_need": {
                "officers": officers,
                "barricades": barricades,
                "signage": signage
            }
        }
