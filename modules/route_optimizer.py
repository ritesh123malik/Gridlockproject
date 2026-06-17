"""
modules/route_optimizer.py
--------------------------
Dynamic route optimization for individual vehicles.
Allows individual motorists or fleet dispatches to compute optimal routes between two junctions
while adjusting corridor edge weights dynamically to bypass congested areas.
"""

import networkx as nx

class RouteOptimizer:
    def __init__(self, rerouting_engine):
        self.rerouting = rerouting_engine
        self.graph = rerouting_engine.base_graph.copy()

    def get_optimal_route(self, start: str, end: str, avoid_corridors: list = None) -> dict:
        """
        Calculates the optimal route between start and end nodes.
        If avoid_corridors is specified, increases edge weights connected to those nodes by 3.0x to discourage routing.
        """
        if start not in self.graph or end not in self.graph:
            return {
                "status": "error",
                "message": f"Start '{start}' or End '{end}' node not found in routing graph.",
                "path": [start, end],
                "distance_km": 0.0,
                "travel_time_min": 0.0
            }

        # Work on a copy of the graph to keep base graph weights intact
        G_temp = self.graph.copy()
        
        if avoid_corridors:
            for c in avoid_corridors:
                if c in G_temp:
                    # Penalize entering this corridor by raising edge weights of all connections to 300%
                    for neighbor in list(G_temp.neighbors(c)):
                        old_w = G_temp[c][neighbor].get("weight", 1.0)
                        G_temp[c][neighbor]["weight"] = old_w * 3.0

        try:
            path = nx.shortest_path(G_temp, source=start, target=end, weight="weight")
            
            # Compute total distance along path using base graph weights
            total_dist = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                total_dist += self.graph[u][v].get("weight", 1.0)
            
            # Speed depends on avoidance penalties; if we path through avoid corridors, transit is slower
            base_speed = 30.0 # km/h
            travel_time_min = round((total_dist / base_speed) * 60, 1)
            
            return {
                "status": "success",
                "path": path,
                "distance_km": round(total_dist, 2),
                "travel_time_min": travel_time_min
            }
            
        except nx.NetworkXNoPath:
            return {
                "status": "error",
                "message": "No routing path available between nodes.",
                "path": [start, end],
                "distance_km": 0.0,
                "travel_time_min": 0.0
            }
