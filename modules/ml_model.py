"""
modules/ml_model.py
-------------------
Hybrid ML Engine. Implements a K-Nearest Neighbors (KNN) regressor in pure NumPy/Pandas.
Trains on a mapped version of ASTRAM historical events (latitude, longitude, corridor, prior impact),
and predicts expected delay, congestion risk, and a prediction confidence level.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from modules.historical import load_historical_events, load_corridor_scores

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EVENT_MAPPINGS = {
    "Political Rally": 0,
    "Festival": 1,
    "Sporting Event": 2,
    "Concert": 3,
    "VIP Movement": 4
}

WEATHER_MAPPINGS = {
    "Sunny": 0,
    "Light Rain": 1,
    "Heavy Rain": 2,
    "Fog": 3
}

LIFECYCLE_MAPPINGS = {
    "Pre-event": 0,
    "During-event": 1,
    "Exit-wave": 2,
    "Post-event": 3
}

class HybridMLPredictor:
    def __init__(self, k: int = 5):
        self.k = k
        self.centroids_df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "corridor_centroids.csv")
        self.corridors = sorted(list(self.centroids_df["corridor"]))
        self.corridor_idx = {name: idx for idx, name in enumerate(self.corridors)}
        
        self.X_train = None
        self.y_delay = None
        self.y_risk = None
        self.feature_means = None
        self.feature_stds = None
        
        self._train_model()

    def _train_model(self):
        """Prepares training data from historical_events.csv and trains the KNN model."""
        hist_events = load_historical_events()
        scores_df = load_corridor_scores()
        scores_lookup = scores_df.set_index("corridor")["impact_score"].to_dict()
        max_score = scores_df["impact_score"].max() if not scores_df.empty else 1.0

        # Subsample to keep computations fast
        sample_df = hist_events.sample(min(800, len(hist_events)), random_state=42).copy()

        X_list = []
        y_delay_list = []
        y_risk_list = []

        # Build synthetic features associated with ASTRAM historical records
        for _, row in sample_df.iterrows():
            corridor = row["corridor"]
            if pd.isna(corridor) or corridor not in self.corridor_idx:
                continue
            
            c_idx = self.corridor_idx[corridor]
            prior = scores_lookup.get(corridor, 0.0) / max_score
            is_obs = int(row["is_obstruction"])
            is_high = 1 if row["priority"] == "High" else 0

            # Map event cause to simulated event types
            cause = str(row["event_cause"]).lower()
            if "public" in cause:
                evt_type = "Political Rally"
            elif "congestion" in cause:
                evt_type = "Festival"
            elif "accident" in cause:
                evt_type = "VIP Movement"
            elif "procession" in cause:
                evt_type = "Sporting Event"
            else:
                evt_type = "Concert"
            
            evt_code = EVENT_MAPPINGS[evt_type]

            # Synthesize crowd size, weather, and lifecycle mapping for training instances
            if is_high == 1:
                crowd = np.random.randint(15000, 35000)
                weather_code = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) # Sunny, Light Rain, Heavy Rain
                lifecycle_code = 1 # During-event
                # Targets: Higher delay & risk
                delay = 25.0 + (crowd / 800.0) + prior * 15.0 + weather_code * 8.0
                risk = 65.0 + (crowd / 400.0) + prior * 10.0
            else:
                crowd = np.random.randint(1000, 8000)
                weather_code = np.random.choice([0, 1], p=[0.85, 0.15])
                lifecycle_code = np.random.choice([0, 1, 3])
                # Targets: lower delay & risk
                delay = 5.0 + (crowd / 1200.0) + prior * 5.0 + weather_code * 3.0
                risk = 15.0 + (crowd / 500.0) + prior * 5.0

            # Vector: [corridor_idx, event_type_idx, crowd_size, weather_idx, lifecycle_idx, prior_score]
            X_list.append([float(c_idx), float(evt_code), float(crowd), float(weather_code), float(lifecycle_code), float(prior)])
            y_delay_list.append(delay)
            y_risk_list.append(risk)

        if not X_list:
            # Fallback training set if empty
            self.X_train = np.zeros((5, 6))
            self.y_delay = np.zeros(5)
            self.y_risk = np.zeros(5)
            self.feature_means = np.zeros(6)
            self.feature_stds = np.ones(6)
            return

        self.X_train = np.array(X_list)
        self.y_delay = np.array(y_delay_list)
        self.y_risk = np.array(y_risk_list)

        # Standardize features for distance calculations
        self.feature_means = self.X_train.mean(axis=0)
        self.feature_stds = self.X_train.std(axis=0)
        # Avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0
        self.X_train_scaled = (self.X_train - self.feature_means) / self.feature_stds

    def predict(self, event_type: str, crowd_size: int, location_corridor: str, 
                weather: str = "Sunny", lifecycle: str = "During-event") -> dict:
        """
        Runs KNN inference using vectorized NumPy.
        Returns predicted delay, risk, and model confidence level.
        """
        if self.X_train is None:
            return {"delay": 10.0, "risk": 50.0, "confidence": 80.0}

        c_idx = self.corridor_idx.get(location_corridor, 0)
        evt_code = EVENT_MAPPINGS.get(event_type, 1)
        weather_code = WEATHER_MAPPINGS.get(weather, 0)
        lifecycle_code = LIFECYCLE_MAPPINGS.get(lifecycle, 1)
        
        # Look up prior score
        scores_df = load_corridor_scores()
        scores_lookup = scores_df.set_index("corridor")["impact_score"].to_dict()
        max_score = scores_df["impact_score"].max() if not scores_df.empty else 1.0
        prior = scores_lookup.get(location_corridor, 0.0) / max_score

        # Query vector
        query = np.array([float(c_idx), float(evt_code), float(crowd_size), float(weather_code), float(lifecycle_code), float(prior)])
        
        # Scale query
        query_scaled = (query - self.feature_means) / self.feature_stds

        # Compute Euclidean distances to all training points
        dists = np.sqrt(np.sum((self.X_train_scaled - query_scaled) ** 2, axis=1))

        # Get top K indices
        nearest_indices = np.argsort(dists)[:self.k]
        nearest_dists = dists[nearest_indices]

        # Predict as averages of nearest neighbors
        pred_delay = float(np.mean(self.y_delay[nearest_indices]))
        pred_risk = float(np.mean(self.y_risk[nearest_indices]))

        # Calculate prediction Confidence Score based on local distance density
        # Average distance to K neighbors (closer neighbors = higher confidence)
        avg_dist = float(np.mean(nearest_dists))
        confidence = 100.0 - (avg_dist * 18.0)
        confidence = max(65.0, min(98.0, confidence))

        return {
            "delay": round(pred_delay, 1),
            "risk": round(pred_risk, 1),
            "confidence": round(confidence, 1)
        }
