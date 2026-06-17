"""
historical.py
--------------
Loads the precomputed Congestion Impact Intelligence derived from real
ASTRAM traffic-event data (Bengaluru, Nov 2023-Apr 2024).

This is the "historical prior" layer: it tells the fusion engine which
corridors and police-station jurisdictions are *already* chronically
congestion-prone, so a live parking-violation detection in one of these
zones gets weighted higher than the same detection on a quiet street.

NOTE ON DATA HONESTY: the source dataset has no labeled "illegal_parking"
field. Scores here are built from vehicle-obstruction-type causes
(vehicle_breakdown, congestion, others, debris) as a physically valid
proxy for "something is occupying the carriageway and disrupting flow" --
the same phenomenon as illegal parking. Document this assumption in your
submission writeup.
"""

from pathlib import Path
import math
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_corridor_scores() -> pd.DataFrame:
    """Corridor-level Congestion Impact Score, sorted descending."""
    df = pd.read_csv(DATA_DIR / "corridor_impact_scores.csv")
    return df.sort_values("impact_score", ascending=False).reset_index(drop=True)


def load_station_scores() -> pd.DataFrame:
    """Police-station-jurisdiction-level Congestion Impact Score."""
    df = pd.read_csv(DATA_DIR / "police_station_impact_scores.csv")
    return df.sort_values("impact_score", ascending=False).reset_index(drop=True)


def load_historical_events() -> pd.DataFrame:
    """Raw (trimmed) historical events, used for the base heatmap layer."""
    return pd.read_csv(DATA_DIR / "historical_events.csv")


def load_corridor_centroids() -> pd.DataFrame:
    """Mean lat/long per named corridor, used for nearest-corridor matching."""
    return pd.read_csv(DATA_DIR / "corridor_centroids.csv")


def load_hourly_counts() -> pd.DataFrame:
    """Raw hour-of-day (IST) event counts, for charting."""
    df = pd.read_csv(DATA_DIR / "hourly_obstruction_pattern_IST.csv")
    df.columns = ["hour", "count"]
    return df


def load_hourly_risk() -> dict:
    """
    Hour-of-day (IST) -> normalized risk weight in [0, 1], built from the
    historical frequency of obstruction-type events at that hour.
    """
    df = load_hourly_counts()
    max_count = df["count"].max()
    return {int(row["hour"]): row["count"] / max_count for _, row in df.iterrows()}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class CorridorMatcher:
    """
    Matches a live detection's (lat, lon) to the nearest named corridor and
    looks up its historical Congestion Impact Score. This is what lets a
    live YOLO detection -- which only has GPS coordinates -- "inherit" the
    historical risk context of the road it's on.
    """

    def __init__(self):
        self.centroids = load_corridor_centroids()
        scores = load_corridor_scores().set_index("corridor")["impact_score"]
        self._score_lookup = scores.to_dict()
        self._max_score = scores.max() if len(scores) else 1.0

    def nearest_corridor(self, lat: float, lon: float, max_km: float = 1.5):
        """
        Returns (corridor_name, distance_km, historical_score_0_to_1) or
        (None, None, 0.0) if nothing is within max_km (i.e. likely a
        non-corridor street -- still valid, just no historical prior).
        """
        best_name, best_dist = None, float("inf")
        for _, row in self.centroids.iterrows():
            d = _haversine_km(lat, lon, row["centroid_lat"], row["centroid_lon"])
            if d < best_dist:
                best_dist, best_name = d, row["corridor"]
        if best_name is None or best_dist > max_km:
            return None, None, 0.0
        raw_score = self._score_lookup.get(best_name, 0.0)
        normalized = raw_score / self._max_score if self._max_score else 0.0
        return best_name, round(best_dist, 3), round(normalized, 3)
