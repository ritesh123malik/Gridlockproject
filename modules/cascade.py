"""
modules/cascade.py
------------------
Incident Cascade Predictor. Forecasts spatial-temporal queue propagation,
time-to-gridlock, and secondary collision risks for neighboring corridors
when an event causes primary congestion.
"""

from modules.historical import load_corridor_centroids

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

class IncidentCascadePredictor:
    def __init__(self):
        self.centroids = load_corridor_centroids()

    def predict_cascade(self, location_corridor: str, base_risk: float, weather: str = "Sunny") -> list[dict]:
        """
        Calculates propagation risk for adjacent corridors.
        """
        # Find target coordinates
        target_row = self.centroids[self.centroids["corridor"] == location_corridor]
        if target_row.empty:
            return []
            
        target_lat = target_row.iloc[0]["centroid_lat"]
        target_lon = target_row.iloc[0]["centroid_lon"]

        rain_factor = 1.35 if weather in ["Light Rain", "Heavy Rain"] else 1.0
        
        cascades = []
        for _, row in self.centroids.iterrows():
            name = row["corridor"]
            if name == location_corridor:
                continue

            dist = _haversine_km(target_lat, target_lon, row["centroid_lat"], row["centroid_lon"])
            # Focus on immediate neighbors within 5.5 km
            if dist <= 5.5:
                # Proximity decay for cascade probability
                prob = base_risk * (1.0 - (dist / 7.0)) * rain_factor
                prob = min(98.0, max(15.0, prob))
                
                # Queue length scales with risk & rain
                queue_meters = int(prob * 18 * rain_factor)
                
                # Time to gridlock (slower propagation if far)
                gridlock_time = max(5, int(dist * 6.0 / rain_factor))
                
                # Secondary accident risk scales with unexpected queue formations
                sec_accident_prob = min(85.0, prob * 0.35 * rain_factor)

                if prob >= 75.0:
                    status_level = "Critical Spillover"
                elif prob >= 50.0:
                    status_level = "High Risk"
                else:
                    status_level = "Moderate Spillover"

                cascades.append({
                    "corridor": name,
                    "distance_km": round(dist, 2),
                    "cascade_risk_pct": round(prob, 1),
                    "queue_length_meters": queue_meters,
                    "time_to_gridlock_mins": gridlock_time,
                    "secondary_accident_risk_pct": round(sec_accident_prob, 1),
                    "status_level": status_level
                })

        # Sort by cascade risk descending
        return sorted(cascades, key=lambda x: x["cascade_risk_pct"], reverse=True)
