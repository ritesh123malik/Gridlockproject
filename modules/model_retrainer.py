"""
modules/model_retrainer.py
--------------------------
Auto-retraining pipeline for the KNN regressor.
Triggers rebuilding model training matrices when enough new logs are logged.
"""

import pandas as pd

class ModelRetrainer:
    def __init__(self, ml_predictor, threshold: int = 5):
        self.ml = ml_predictor
        self.threshold = threshold

    def should_retrain(self, feedback_df: pd.DataFrame) -> bool:
        """
        Returns True if the feedback dataset contains at least the threshold count.
        """
        if feedback_df is None or feedback_df.empty:
            return False
        return len(feedback_df) >= self.threshold

    def retrain(self) -> bool:
        """
        Triggers retraining logic inside the HybridMLPredictor instance.
        """
        try:
            self.ml.retrain()
            return True
        except Exception as e:
            print(f"Error during retraining execution: {e}")
            return False
