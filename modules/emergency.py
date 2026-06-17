"""
modules/emergency.py
--------------------
Emergency priority route protection. Recommends dedicated Green Corridors,
intersection signal overrides, and heavy vehicle blocking zones for emergency vehicle routing.
"""

from modules.rerouting import ReroutingEngine

class EmergencyPriorityEngine:
    def __init__(self, rerouting_engine: ReroutingEngine):
        self.rerouting = rerouting_engine

    def generate_green_corridor(self, start_node: str, end_node: str, blocked_node: str) -> dict:
        """
        Inputs:
          - start_node: corridor start point
          - end_node: corridor destination
          - blocked_node: event congestion node to bypass
        Returns:
          - green_corridor_path: list of nodes
          - signal_overrides: list of override commands
          - lane_restrictions: list of vehicle blocks
        """
        # Calculate detours using ReroutingEngine
        detours = self.rerouting.calculate_detours(start_node, end_node, blocked_node)
        
        if detours.get("status") != "success":
            return {
                "status": "error",
                "message": detours.get("message", "Routing calculation failed")
            }

        emergency_path = detours["emergency_route"]["nodes"]
        
        signal_overrides = []
        lane_restrictions = []

        # Generate overrides for junctions on the emergency route
        for i, node in enumerate(emergency_path):
            signal_overrides.append(
                f"🎛️ Junction Override: Hold GREEN phase at **{node} Junction** for emergency convoy approach."
            )
            
            # Place entry blocks for heavy vehicles on intersections
            if i > 0 and i < len(emergency_path) - 1:
                lane_restrictions.append(
                    f"🚫 Heavy Vehicle Block: Bar entry of commercial trucks onto **{node}** from adjoining radials."
                )

        # General directives
        lane_restrictions.append(
            f"⚡ Lane Segregation: Deploy police escorts on **{', '.join(emergency_path[:3])}** service lanes to establish a clear medical lane."
        )

        return {
            "status": "success",
            "green_corridor_path": emergency_path,
            "distance_km": detours["emergency_route"]["distance_km"],
            "travel_time_min": detours["emergency_route"]["travel_time_min"],
            "signal_overrides": signal_overrides,
            "lane_restrictions": lane_restrictions
        }
