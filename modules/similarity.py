"""
modules/similarity.py
---------------------
Search engine to find the most similar historical major events in Bengaluru
and extract their outcomes, actions taken, and lessons learned to provide
practical decision support for traffic planners.
"""

import math
import pandas as pd
from pathlib import Path
from modules.historical import load_corridor_centroids

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

class EventSimilaritySearch:
    def __init__(self):
        self.centroids = load_corridor_centroids()
        self.major_events_file = DATA_DIR / "historical_major_events.csv"
        if self.major_events_file.exists():
            self.events = pd.read_csv(self.major_events_file)
        else:
            self.events = pd.DataFrame(columns=[
                "event_type", "location", "crowd_size", "actual_delay",
                "actual_risk", "actions_taken", "lessons_learned"
            ])

    def find_similar_events(self, event_type: str, crowd_size: int, location_corridor: str, top_n: int = 3) -> list[dict]:
        """
        Calculates similarity using:
          - Event Type match (40% weight)
          - Crowd Size ratio (30% weight)
          - Corridor Distance proximity (30% weight)
        """
        if self.events.empty:
            return []

        # Target coordinates
        target_row = self.centroids[self.centroids["corridor"] == location_corridor]
        target_lat = target_row.iloc[0]["centroid_lat"] if not target_row.empty else 12.9716
        target_lon = target_row.iloc[0]["centroid_lon"] if not target_row.empty else 77.5946

        results = []
        for _, row in self.events.iterrows():
            # 1. Type similarity
            type_sim = 1.0 if row["event_type"].lower() == event_type.lower() else 0.2

            # 2. Crowd similarity (using normalized ratio)
            c1, c2 = float(row["crowd_size"]), float(crowd_size)
            crowd_sim = 1.0 - (abs(c1 - c2) / max(c1, c2, 1.0))

            # 3. Location distance similarity
            hist_row = self.centroids[self.centroids["corridor"] == row["location"]]
            if not hist_row.empty:
                hist_lat = hist_row.iloc[0]["centroid_lat"]
                hist_lon = hist_row.iloc[0]["centroid_lon"]
                dist = _haversine_km(target_lat, target_lon, hist_lat, hist_lon)
            else:
                dist = 15.0 # Max boundary penalty
            
            loc_sim = max(0.0, 1.0 - (dist / 15.0)) # decays to 0 at 15km

            # Composite score (0 to 1.0)
            score = 0.4 * type_sim + 0.3 * crowd_sim + 0.3 * loc_sim
            similarity_pct = round(score * 100, 1)

            results.append({
                "event_type": row["event_type"],
                "location": row["location"],
                "crowd_size": int(row["crowd_size"]),
                "actual_delay": int(row["actual_delay"]),
                "actual_risk": int(row["actual_risk"]),
                "actions_taken": row["actions_taken"],
                "lessons_learned": row["lessons_learned"],
                "similarity_pct": similarity_pct,
                "distance_km": round(dist, 1) if not hist_row.empty else "N/A"
            })

        # Sort by similarity descending
        results = sorted(results, key=lambda x: x["similarity_pct"], reverse=True)
        return results[:top_n]
