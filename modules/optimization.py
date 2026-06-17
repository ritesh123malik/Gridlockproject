"""
modules/optimization.py
-----------------------
Constrained Resource Optimization. Enforces station-level officer capacity
constraints (e.g. Hebbal max 12 officers, Bellandur max 15) dynamically matched
to corridor jurisdictions to prevent over-allocating officers beyond a station's pool.
"""

# Map corridors to police stations and their maximum available officer caps
STATION_JURISDICTIONS = {
    "Bellary Road 1": ("Sadashivanagar PS", 12),
    "Bellary Road 2": ("Yelahanka PS", 10),
    "ORR East 1": ("Bellandur PS", 15),
    "ORR East 2": ("Mahadevapura PS", 15),
    "Mysore Road": ("Byatarayanapura PS", 12),
    "Hosur Road": ("Madiwala PS", 15),
    "Bannerghata Road": ("Mico Layout PS", 12),
    "Tumkur Road": ("Yeshwanthpura PS", 10),
    "Magadi Road": ("Kamakshipalya PS", 8),
    "Old Madras Road": ("Halasur PS", 12),
    "CBD 2": ("Cubbon Park PS", 8),
    "CBD 1": ("Shivajinagar PS", 8),
    "West of Chord Road": ("Vijayanagara PS", 8),
    "Old Airport Road": ("HAL Old Airport PS", 10),
    "Hennur Main Road": ("Hennuru PS", 10),
    "IRR(Thanisandra road)": ("Jeevanbheemanagar PS", 8),
    "Varthur Road": ("Whitefield PS", 10),
    "Airport New South Road": ("Banaswadi PS", 8),
}

class PatrolOptimizer:
    def __init__(self, beta: float = 0.15):
        self.beta = beta

    def get_station_info(self, zone_name: str) -> tuple[str, int]:
        """Returns (police_station_name, max_capacity)."""
        return STATION_JURISDICTIONS.get(zone_name, ("General Standby PS", 6))

    def optimize_officers(self, affected_zones_risk: dict[str, float], total_officers: int) -> dict:
        """
        Inputs:
          - affected_zones_risk: dict of {zone_name: base_congestion_risk_percentage}
          - total_officers: total pool of officers available to allocate globally
        Enforces Station-Level capacity caps while greedily optimizing, calculating deficits and borrowing plans.
        """
        # Calculate required officers based on risk points
        required_officers = sum(max(2, int(r * 0.15)) for r in affected_zones_risk.values()) if affected_zones_risk else 0
        deficit = max(0, required_officers - total_officers)

        # Active stations in this incident
        active_stations = {self.get_station_info(z)[0] for z in affected_zones_risk} if affected_zones_risk else set()

        # Compile borrowing plan from inactive stations
        borrowing_plan = []
        if deficit > 0:
            inactive_stations = {}
            for corridor, (station, cap) in STATION_JURISDICTIONS.items():
                if station not in active_stations:
                    inactive_stations[station] = max(inactive_stations.get(station, 0), cap)
            
            sorted_inactive = sorted(inactive_stations.items(), key=lambda x: x[1], reverse=True)
            borrowed = 0
            for station, cap in sorted_inactive:
                if borrowed >= deficit:
                    break
                needed = deficit - borrowed
                to_borrow = min(needed, cap)
                if to_borrow > 0:
                    borrowing_plan.append({"station": station, "borrow_count": to_borrow})
                    borrowed += to_borrow

        if not affected_zones_risk or total_officers <= 0:
            return {
                "allocations": {z: 0 for z in affected_zones_risk},
                "details": [{"zone": z, "station": self.get_station_info(z)[0], "base_risk": r, "final_risk": r, "officers": 0, "capacity_limit": self.get_station_info(z)[1]} for z, r in affected_zones_risk.items()],
                "network_risk_before": sum(affected_zones_risk.values()) / max(1, len(affected_zones_risk)) if affected_zones_risk else 0.0,
                "network_risk_after": sum(affected_zones_risk.values()) / max(1, len(affected_zones_risk)) if affected_zones_risk else 0.0,
                "required_officers": required_officers,
                "deficit": deficit,
                "borrowing_plan": borrowing_plan
            }

        # Initialize allocations
        allocations = {zone: 0 for zone in affected_zones_risk}
        
        # Track active station allocation sums
        station_allocations = {}

        def get_risk(zone, x):
            base_risk = affected_zones_risk[zone]
            return base_risk / (1.0 + self.beta * x)

        # Greedy allocation with constraints
        for _ in range(total_officers):
            best_zone = None
            max_benefit = -1.0
            
            for zone in affected_zones_risk:
                station_name, capacity = self.get_station_info(zone)
                current_station_alloc = station_allocations.get(station_name, 0)
                
                # ENFORCE CONSTRAINT: If station capacity is saturated, skip this zone!
                if current_station_alloc >= capacity:
                    continue
                
                current_alloc = allocations[zone]
                current_risk = get_risk(zone, current_alloc)
                next_risk = get_risk(zone, current_alloc + 1)
                benefit = current_risk - next_risk
                
                if benefit > max_benefit:
                    max_benefit = benefit
                    best_zone = zone
            
            if best_zone is not None:
                allocations[best_zone] += 1
                station_name, _ = self.get_station_info(best_zone)
                station_allocations[station_name] = station_allocations.get(station_name, 0) + 1
            else:
                # All eligible stations are saturated, stop allocating early
                break

        # Compile detailed matrix
        details = []
        total_risk_before = 0.0
        total_risk_after = 0.0
        
        for zone, base_risk in affected_zones_risk.items():
            num_officers = allocations[zone]
            final_risk = get_risk(zone, num_officers)
            station_name, capacity = self.get_station_info(zone)
            
            total_risk_before += base_risk
            total_risk_after += final_risk
            
            details.append({
                "zone": zone,
                "station": station_name,
                "base_risk": round(base_risk, 1),
                "final_risk": round(final_risk, 1),
                "officers": num_officers,
                "capacity_limit": capacity,
                "risk_reduction_pct": round(base_risk - final_risk, 1)
            })

        n_zones = len(affected_zones_risk)
        return {
            "allocations": allocations,
            "details": sorted(details, key=lambda x: x["officers"], reverse=True),
            "network_risk_before": round(total_risk_before / n_zones, 1),
            "network_risk_after": round(total_risk_after / n_zones, 1),
            "required_officers": required_officers,
            "deficit": deficit,
            "borrowing_plan": borrowing_plan
        }
