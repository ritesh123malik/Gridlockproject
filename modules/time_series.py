"""
modules/time_series.py
-----------------------
Time-series congestion forecasting using Holt's double exponential smoothing.
No external dependencies beyond numpy — runs on the existing .venv.
Predicts next 24 hourly congestion counts from historical ASTRAM data.
"""

import numpy as np


class HourlyForecaster:
    def __init__(self, alpha: float = 0.4, beta: float = 0.3):
        """
        alpha — smoothing factor for level (0 < alpha < 1).
        beta  — smoothing factor for trend (0 < beta < 1).
        Higher alpha reacts faster to recent changes; higher beta tracks trends faster.
        """
        self.alpha = alpha
        self.beta = beta

    def _holt_smooth(self, series: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Applies Holt's double exponential smoothing to the input series.
        Returns (smoothed_series, final_level, final_trend).
        """
        n = len(series)
        level = series[0]
        trend = series[1] - series[0] if n > 1 else 0.0
        smoothed = []
        for t in range(n):
            if t == 0:
                smoothed.append(level)
                continue
            prev_level = level
            level = self.alpha * series[t] + (1 - self.alpha) * (prev_level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            smoothed.append(level)
        return np.array(smoothed), level, trend

    def forecast_24h(self, hourly_df) -> dict:
        """
        Accepts a DataFrame with 'hour' and 'count' columns (the hourly ASTRAM data).
        Returns dict with:
          - historical_hours: list[int]
          - historical_counts: list[float]
          - forecast_hours: list[int]  (24→47 in 24-hour extension)
          - forecast_counts: list[float]
          - upper_bound: list[float]   (95% confidence band)
          - lower_bound: list[float]
        """
        counts = hourly_df.sort_values("hour")["count"].values.astype(float)
        hours = hourly_df.sort_values("hour")["hour"].values.tolist()

        smoothed, final_level, final_trend = self._holt_smooth(counts)

        # Project 24 hours forward
        forecast = []
        for h in range(1, 25):
            val = max(0, final_level + h * final_trend)
            # Add sinusoidal daily pattern to replicate rush-hour shape
            hour_of_day = (hours[-1] + h) % 24
            rush_factor = 1.0 + 0.35 * np.sin(np.pi * (hour_of_day - 7) / 8) \
                              + 0.25 * np.sin(np.pi * (hour_of_day - 17) / 6)
            rush_factor = max(0.3, rush_factor)
            forecast.append(round(val * rush_factor, 1))

        # Confidence band based on residual std
        residuals = counts - smoothed
        std_err = np.std(residuals)
        upper = [round(f + 1.96 * std_err * (i + 1) ** 0.5, 1) for i, f in enumerate(forecast)]
        lower = [round(max(0, f - 1.96 * std_err * (i + 1) ** 0.5), 1) for i, f in enumerate(forecast)]

        forecast_hours = [(hours[-1] + h) % 24 for h in range(1, 25)]
        forecast_labels = [f"{h:02d}:00 (+{h}h)" for h in range(1, 25)]

        return {
            "historical_hours": hours,
            "historical_counts": counts.tolist(),
            "forecast_hours": forecast_hours,
            "forecast_labels": forecast_labels,
            "forecast_counts": forecast,
            "upper_bound": upper,
            "lower_bound": lower
        }
