"""
modules/sentiment.py
--------------------
Simulated public sentiment based on event parameters and weather.
"""

class SentimentAnalyzer:
    def analyze(self, event_type: str, crowd_size: int, weather: str = "Sunny") -> dict:
        """
        Estimates public mood, positive rating score, and social tweet volumes.
        Penalizes sentiment score for large crowds and wet weather conditions.
        """
        base = 0.75
        
        # Penalize for massive crowds (scale penalty)
        crowd_penalty = crowd_size / 120000.0
        
        # Penalize for poor weather
        weather_penalty = 0.0
        if weather == "Heavy Rain":
            weather_penalty = 0.15
        elif weather == "Light Rain" or weather == "Fog":
            weather_penalty = 0.05
            
        score = max(0.05, base - crowd_penalty - weather_penalty)
        score = round(score, 2)
        
        if score > 0.6:
            mood = "Positive 😊"
        elif score > 0.4:
            mood = "Neutral 😐"
        else:
            mood = "Negative 😠"
            
        # Social media engagement rate: ~3-5% of event attendance tweets
        estimated_tweets = int(crowd_size * 0.04)
        
        return {
            "sentiment_score": score,
            "mood": mood,
            "estimated_tweets": estimated_tweets
        }
