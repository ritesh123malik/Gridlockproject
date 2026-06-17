"""
modules/rerouting.py
--------------------
Uses networkx to build a logical topological road network graph of Bengaluru corridors.
If an event blocks a corridor, computes:
  - Route A: Primary detour shortest path bypassing the block.
  - Route B: Secondary detour shortest path.
  - Emergency Route: Fastest route, prioritizing major Ring Roads (ORR) and radial expressways.
"""

import math
import networkx as nx
from modules.historical import load_corridor_centroids

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

# Logical connections in Bengaluru corridor network
CORRIDOR_EDGES = [
    ("Tumkur Road", "West of Chord Road"),
    ("Tumkur Road", "ORR North 2"),
    ("West of Chord Road", "Mysore Road"),
    ("West of Chord Road", "Magadi Road"),
    ("Magadi Road", "CBD 2"),
    ("Mysore Road", "CBD 2"),
    ("Mysore Road", "ORR West 1"),
    ("ORR West 1", "Bannerghata Road"),
    ("Bannerghata Road", "Hosur Road"),
    ("Hosur Road", "ORR East 1"),
    ("ORR East 1", "ORR East 2"),
    ("ORR East 2", "Old Airport Road"),
    ("ORR East 2", "Varthur Road"),
    ("ORR East 2", "Old Madras Road"),
    ("Varthur Road", "Old Airport Road"),
    ("Old Airport Road", "CBD 1"),
    ("Old Madras Road", "CBD 1"),
    ("Old Madras Road", "ORR North 1"),
    ("ORR North 1", "Airport New South Road"),
    ("ORR North 1", "Hennur Main Road"),
    ("Hennur Main Road", "IRR(Thanisandra road)"),
    ("Hennur Main Road", "Bellary Road 1"),
    ("IRR(Thanisandra road)", "Hosur Road"),
    ("Bellary Road 1", "Bellary Road 2"),
    ("Bellary Road 1", "ORR North 2"),
    ("Bellary Road 1", "CBD 2"),
    ("CBD 2", "CBD 1"),
]

class ReroutingEngine:
    def __init__(self):
        self.centroids = load_corridor_centroids()
        self.centroids_dict = self.centroids.set_index("corridor").to_dict(orient="index")
        self.base_graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        # Add nodes with coords
        for corridor, data in self.centroids_dict.items():
            G.add_node(corridor, lat=data["centroid_lat"], lon=data["centroid_lon"])

        # Add logical edges with haversine distance
        for u, v in CORRIDOR_EDGES:
            if u in G.nodes and v in G.nodes:
                lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
                lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
                dist = _haversine_km(lat1, lon1, lat2, lon2)
                G.add_edge(u, v, weight=dist, capacity="regular")
        
        # Connect isolated components just in case (ensure connectivity)
        components = list(nx.connected_components(G))
        if len(components) > 1:
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i+1])[0]
                lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
                lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
                dist = _haversine_km(lat1, lon1, lat2, lon2)
                G.add_edge(u, v, weight=dist, capacity="regular")

        return G

    def get_corridor_coordinates(self, corridor_name: str) -> tuple:
        if corridor_name in self.centroids_dict:
            c = self.centroids_dict[corridor_name]
            return c["centroid_lat"], c["centroid_lon"]
        return 12.9716, 77.5946

    def calculate_detours(self, start_node: str, end_node: str, blocked_node: str) -> dict:
        """
        Calculates Route A, Route B and Emergency Route bypassing blocked_node.
        Returns paths with coordinates, cumulative distance, and estimated travel time.
        """
        if start_node not in self.base_graph or end_node not in self.base_graph:
            return {"status": "error", "message": "Start/End corridor not found"}

        # Graph for regular detour (Route A & B) -- remove the blocked node
        G_detour = self.base_graph.copy()
        if blocked_node in G_detour:
            # If start or end is blocked, we can't route, but if it's intermediate we bypass it
            if start_node != blocked_node and end_node != blocked_node:
                G_detour.remove_node(blocked_node)

        # Route A (shortest path)
        try:
            path_a = nx.shortest_path(G_detour, source=start_node, target=end_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_a = [start_node, end_node]  # Fallback direct

        # Route B (alternative path, by removing edges of path_a)
        G_alt = G_detour.copy()
        if len(path_a) > 2:
            # Remove an edge from path_a to force a different path
            mid_idx = len(path_a) // 2
            u, v = path_a[mid_idx - 1], path_a[mid_idx]
            if G_alt.has_edge(u, v):
                G_alt.remove_edge(u, v)
        try:
            path_b = nx.shortest_path(G_alt, source=start_node, target=end_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_b = path_a  # Fallback

        # Emergency Route: prefers high-capacity Ring Roads (ORR)
        # We modify the detour graph weights: ORR roads get 40% weight reduction
        G_emergency = G_detour.copy()
        for u, v, d in G_emergency.edges(data=True):
            if "ORR" in u or "ORR" in v or "Airport" in u or "Airport" in v:
                d["weight"] = d["weight"] * 0.4 # Prefer ring roads

        try:
            path_e = nx.shortest_path(G_emergency, source=start_node, target=end_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_e = path_a

        # Format details for plotting
        def format_path(nodes_list, speed_kmh=30):
            coords = []
            total_dist = 0.0
            prev_lat, prev_lon = None, None
            for node in nodes_list:
                lat, lon = self.get_corridor_coordinates(node)
                coords.append({"name": node, "lat": lat, "lon": lon})
                if prev_lat is not None:
                    total_dist += _haversine_km(prev_lat, prev_lon, lat, lon)
                prev_lat, prev_lon = lat, lon
            
            # Delay estimate
            travel_time_min = round((total_dist / speed_kmh) * 60, 1)
            return {
                "nodes": nodes_list,
                "coordinates": coords,
                "distance_km": round(total_dist, 2),
                "travel_time_min": travel_time_min
            }

        return {
            "status": "success",
            "route_a": format_path(path_a, speed_kmh=25), # regular detour speed
            "route_b": format_path(path_b, speed_kmh=22), # slightly slower
            "emergency_route": format_path(path_e, speed_kmh=45) # fast emergency wave speed
        }

    def get_route_edges(self, path: list) -> list[tuple]:
        """
        Returns the list of edges (u, v) for a given path sequence of nodes.
        """
        if not path or len(path) < 2:
            return []
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]

    def apply_live_delay_weights(self, corridor_delays: dict):
        """
        Adjusts the edge weights in the graph dynamically based on live delays.
        corridor_delays: dict of {corridor_name: delay_minutes}
        """
        # Reset edge weights first
        for u, v, d in self.base_graph.edges(data=True):
            lat1, lon1 = self.base_graph.nodes[u]["lat"], self.base_graph.nodes[u]["lon"]
            lat2, lon2 = self.base_graph.nodes[v]["lat"], self.base_graph.nodes[v]["lon"]
            dist = _haversine_km(lat1, lon1, lat2, lon2)
            d["weight"] = dist
            
        # Apply delays
        for corridor, delay in corridor_delays.items():
            if corridor in self.base_graph:
                penalty_factor = 1.0 + (delay / 10.0) # E.g. 10m delay doubles weight
                for neighbor in list(self.base_graph.neighbors(corridor)):
                    self.base_graph[corridor][neighbor]["weight"] *= penalty_factor
