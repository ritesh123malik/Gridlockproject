"""
modules/learning.py
-------------------
Post-Event Learning System. Logs forecasting predictions against actual outcomes
to calculate system bias and adjust future model predictions dynamically.
Seeded with historical feedback data so the system displays calibration states immediately.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED_FEEDBACK = [
    {"timestamp": "2026-06-14 10:30", "event_type": "Festival", "location": "Mysore Road", "crowd_size": 18000, "predicted_delay": 30.0, "actual_delay": 32.0, "predicted_risk": 80.0, "actual_risk": 82.0, "difference": 2.0},
    {"timestamp": "2026-06-15 18:45", "event_type": "Concert", "location": "ORR East 1", "crowd_size": 15000, "predicted_delay": 25.0, "actual_delay": 28.0, "predicted_risk": 72.0, "actual_risk": 75.0, "difference": 3.0},
    {"timestamp": "2026-06-16 11:15", "event_type": "Political Rally", "location": "Bellary Road 1", "crowd_size": 22000, "predicted_delay": 45.0, "actual_delay": 48.0, "predicted_risk": 95.0, "actual_risk": 96.0, "difference": 3.0},
    {"timestamp": "2026-06-17 09:00", "event_type": "VIP Movement", "location": "Airport New South Road", "crowd_size": 5000, "predicted_delay": 14.0, "actual_delay": 15.0, "predicted_risk": 65.0, "actual_risk": 68.0, "difference": 1.0},
]

class PostEventLearningSystem:
    def __init__(self):
        self.filepath = DATA_DIR / "post_event_feedback.csv"
        self._init_db()

    def _init_db(self):
        if not self.filepath.exists():
            df = pd.DataFrame(SEED_FEEDBACK)
            df.to_csv(self.filepath, index=False)

    def log_event_feedback(self, event_type: str, location: str, crowd_size: int, 
                           predicted_delay: float, actual_delay: float, 
                           predicted_risk: float, actual_risk: float) -> dict:
        """Logs prediction vs actual data and calculates the error."""
        diff = round(actual_delay - predicted_delay, 1)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "event_type": event_type,
            "location": location,
            "crowd_size": crowd_size,
            "predicted_delay": round(predicted_delay, 1),
            "actual_delay": round(actual_delay, 1),
            "predicted_risk": round(predicted_risk, 1),
            "actual_risk": round(actual_risk, 1),
            "difference": diff
        }
        
        # Load, append, save
        df = pd.read_csv(self.filepath)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.filepath, index=False)
        return row

    def get_history(self) -> pd.DataFrame:
        """Returns the full logs dataframe."""
        return pd.read_csv(self.filepath)

    def get_correction_factor(self) -> float:
        """
        Calculates a calibration correction factor.
        Method: Average Relative Error = mean((actual_delay - predicted_delay) / predicted_delay)
        """
        df = self.get_history()
        if df.empty or len(df) < 2:
            return 0.0
        
        # Compute relative error for each row
        relative_errors = (df["actual_delay"] - df["predicted_delay"]) / df["predicted_delay"].replace(0, 1)
        mean_relative_error = relative_errors.mean()
        
        # Cap correction between -50% and +100% to keep forecast stable
        return max(-0.5, min(1.0, float(mean_relative_error)))

    def reset_learning(self):
        """Resets the learning log to seed data."""
        df = pd.DataFrame(SEED_FEEDBACK)
        df.to_csv(self.filepath, index=False)

    def get_feedback_count(self) -> int:
        """Returns the total number of feedback records stored."""
        df = self.get_history()
        return len(df)

    def trigger_retraining(self, forecaster, threshold: int = 5) -> bool:
        """
        Creates a ModelRetrainer, monitors feedback count, and retrains if threshold is met.
        """
        from modules.model_retrainer import ModelRetrainer
        retrainer = ModelRetrainer(forecaster.ml_predictor, threshold=threshold)
        df = self.get_history()
        if retrainer.should_retrain(df):
            return retrainer.retrain()
        return False
