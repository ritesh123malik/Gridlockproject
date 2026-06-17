"""
modules/multi_event.py
-----------------------
Multi-event simultaneous simulation engine.
Combines two concurrent events and computes merged corridor risk, vehicle totals,
resource conflicts, and combined officer requirements.
"""


class MultiEventSimulator:
    def __init__(self, forecaster):
        self.forecaster = forecaster

    def simulate(self, event1: dict, event2: dict) -> dict:
        """
        Runs forecasts for two events and merges their impacts.
        event1/event2: dicts with keys:
          event_type, crowd_size, location_corridor, weather, lifecycle
        Returns combined stats and conflict analysis.
        """
        r1 = self.forecaster.predict_impact(
            event1["event_type"], event1["crowd_size"],
            event1["location_corridor"], event1["weather"], event1["lifecycle"]
        )
        r2 = self.forecaster.predict_impact(
            event2["event_type"], event2["crowd_size"],
            event2["location_corridor"], event2["weather"], event2["lifecycle"]
        )

        # Build merged corridor risk dict (weighted sum, capped at 99)
        merged_risks = {}
        corridors_1 = {c["corridor"]: c for c in r1["affected_corridors"]}
        corridors_2 = {c["corridor"]: c for c in r2["affected_corridors"]}

        all_corridors = set(corridors_1.keys()) | set(corridors_2.keys())
        conflict_corridors = set(corridors_1.keys()) & set(corridors_2.keys())

        for corr in all_corridors:
            risk1 = r1["congestion_risk"] * (1 - 0.05 * corridors_1.get(corr, {}).get("distance_km", 5)) if corr in corridors_1 else 0
            risk2 = r2["congestion_risk"] * (1 - 0.05 * corridors_2.get(corr, {}).get("distance_km", 5)) if corr in corridors_2 else 0
            combined = min(99.0, risk1 + risk2 * 0.6)  # non-linear merge
            merged_risks[corr] = round(max(0, combined), 1)

        # Combined totals
        total_vehicles = r1["vehicles_generated"] + r2["vehicles_generated"]
        total_delay = round(max(r1["expected_delay"], r2["expected_delay"]) * 1.3, 1)
        total_officers = r1["resource_need"]["officers"] + r2["resource_need"]["officers"]
        combined_loss = round(r1["economics"]["economic_loss_lakhs"] + r2["economics"]["economic_loss_lakhs"], 2)

        # Conflict severity
        conflict_severity = "CRITICAL" if len(conflict_corridors) >= 3 else \
                            "HIGH" if len(conflict_corridors) >= 1 else "MANAGEABLE"

        return {
            "event1": {
                "type": event1["event_type"],
                "location": event1["location_corridor"],
                "risk": r1["congestion_risk"],
                "delay": r1["expected_delay"]
            },
            "event2": {
                "type": event2["event_type"],
                "location": event2["location_corridor"],
                "risk": r2["congestion_risk"],
                "delay": r2["expected_delay"]
            },
            "merged_corridor_risks": merged_risks,
            "conflict_corridors": list(conflict_corridors),
            "conflict_severity": conflict_severity,
            "combined_vehicles": total_vehicles,
            "combined_delay": total_delay,
            "combined_officers_needed": total_officers,
            "combined_economic_loss_lakhs": combined_loss
        }
