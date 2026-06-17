"""
modules/signal_opt.py
--------------------
Adaptive Signal Timing Optimization. Calculates specific green/red timing offset plans
for junctions around an event corridor using "entry volume gating" and "exit volume flushing" rules.
"""

class AdaptiveSignalOptimizer:
    def __init__(self):
        # Coordinates standard junction names for major corridors
        self.junction_map = {
            "Bellary Road 1": "Hebbal Flyover Cross",
            "Bellary Road 2": "Yelahanka Bypass Junction",
            "ORR East 1": "Bellandur Iblur Circle",
            "ORR East 2": "Mahadevapura Outer Ring Road Cross",
            "Mysore Road": "Nayandahalli Metro Junction",
            "Hosur Road": "Central Silk Board Intersect",
            "Bannerghata Road": "Dairy Circle Junction",
            "Tumkur Road": "Goraguntepalya Circle",
            "Magadi Road": "Kamakshipalya Junction",
            "Old Madras Road": "K.R. Puram Hanging Bridge",
            "CBD 2": "Hudson Circle",
            "CBD 1": "MG Road Junction",
            "West of Chord Road": "Rajajinagar 1st Block Cross",
            "Old Airport Road": "HAL Old Airport Road Circle"
        }
        from modules.signal_controller import AdaptiveSignalController
        self.rl_controller = AdaptiveSignalController()

    def optimize_signal_timings(self, location_corridor: str, affected_corridors: list[str], 
                                 base_delay_mins: float) -> list[dict]:
        """
        Calculates signal offsets:
          - Congestion Source: positive offset (+Green) to FLUSH traffic.
          - Inbound feeders: negative offset (-Green) to GATE/THROTTLE incoming flow.
          - Detour/Alternative pathways: positive offset (+Green) to clear detours.
        Incorporates Q-learning actions from reinforcement learning controller.
        """
        plan = []
        
        # Determine traffic state for RL model
        state = "HIGH_CONGESTION" if base_delay_mins > 25.0 else ("MODERATE_CONGESTION" if base_delay_mins > 10.0 else "LOW_CONGESTION")
        rl_action = self.rl_controller.get_action(state)
        from modules.signal_controller import SIGNAL_ACTIONS
        rl_desc = SIGNAL_ACTIONS.get(rl_action, "Maintain Phase (0s)")

        # 1. Congestion source junction
        source_junction = self.junction_map.get(location_corridor, f"{location_corridor} Intersection")
        flush_offset = min(30, int(base_delay_mins * 0.6 + 10))
        plan.append({
            "junction": source_junction,
            "corridor": location_corridor,
            "type": "Exit Flush",
            "offset_seconds": flush_offset,
            "recommended_action": f"Inject **+{flush_offset}s GREEN wave** on main exit lanes to flush stationary queues. [RL Suggests: {rl_desc}]",
            "current_cycle_sec": 120,
            "rl_action_desc": rl_desc
        })

        # 2. Inbound feeder junctions (first 2 affected neighbors - throttle them)
        feeders = [c for c in affected_corridors if c != location_corridor][:2]
        for f in feeders:
            f_junction = self.junction_map.get(f, f"{f} Intersection")
            gate_offset = -max(6, min(15, int(base_delay_mins * 0.3)))
            plan.append({
                "junction": f_junction,
                "corridor": f,
                "type": "Entry Gating",
                "offset_seconds": gate_offset,
                "recommended_action": f"Apply **{gate_offset}s GREEN reduction** on lanes heading towards {location_corridor} to throttle volume. [RL Suggests: {rl_desc}]",
                "current_cycle_sec": 120,
                "rl_action_desc": rl_desc
            })

        # 3. Detour alternative paths (remaining affected neighbors - clear them)
        detours = [c for c in affected_corridors if c != location_corridor][2:4]
        for d in detours:
            d_junction = self.junction_map.get(d, f"{d} Intersection")
            detour_offset = max(8, min(20, int(base_delay_mins * 0.4)))
            plan.append({
                "junction": d_junction,
                "corridor": d,
                "type": "Detour Wave",
                "offset_seconds": detour_offset,
                "recommended_action": f"Synchronize **+{detour_offset}s green offsets** along detour routes to absorb bypass spillover. [RL Suggests: {rl_desc}]",
                "current_cycle_sec": 120,
                "rl_action_desc": rl_desc
            })

        return plan
