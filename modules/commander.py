"""
modules/commander.py
---------------------
AI Traffic Commander Engine. Converts raw predictive data, optimal patrol allocations,
and detour computations into a structured Operational Command Order, providing
decisions rather than just analytics.
"""

class AITrafficCommander:
    def __init__(self):
        pass

    def generate_command_order(self, event_type: str, location: str, crowd_size: int,
                               forecast: dict, allocations: dict, detours: dict,
                               active_diversions: int = 0, signal_adjustment: int = 0) -> dict:
        """
        Synthesizes operational orders based on input forecasts, optimized allocations, and routing.
        """
        officers_needed = forecast["resource_need"]["officers"]
        barricades_needed = forecast["resource_need"]["barricades"]
        
        # Apply reductions from active What-If simulation items
        # If user added diversions, we reduce officer stress slightly or note it in the order
        mitigated_officers = max(3, officers_needed - active_diversions * 2)
        mitigated_barricades = max(2, barricades_needed - active_diversions)

        # Retrieve Route A nodes for detailing the diversion
        detour_a_nodes = []
        if detours.get("status") == "success":
            detour_a_nodes = detours["route_a"]["nodes"]

        # Build specific command directives
        patrol_directives = []
        for zone, num in allocations.items():
            if num > 0:
                patrol_directives.append(f"Deploy **{num} officers** to **{zone}** corridor junctions.")

        if not patrol_directives:
            patrol_directives.append(f"Deploy standard standby patrol of **{mitigated_officers} officers** to **{location}**.")

        # Reconstruct route recommendations
        diversion_directives = []
        if len(detour_a_nodes) > 1:
            bypass_text = " -> ".join(detour_a_nodes)
            diversion_directives.append(f"Activate Detour **Route A** to bypass **{location}**: `{bypass_text}`")
            if detours.get("route_b") and len(detours["route_b"]["nodes"]) > 1:
                alt_text = " -> ".join(detours["route_b"]["nodes"])
                diversion_directives.append(f"Prepare **Route B** as secondary spillover: `{alt_text}`")
        else:
            diversion_directives.append(f"Establish localized detour lanes adjacent to **{location}** junctions.")

        # Special tactical barricades positions
        junction_names = {
            "Bellary Road 1": "Hebbal Flyover Junction",
            "ORR East 1": "Bellandur Outer Ring Road Cross",
            "Mysore Road": "Nayandahalli Junction",
            "Tumkur Road": "Peenya Metro Station Circle",
            "CBD 2": "Hudson Circle",
            "CBD 1": "MG Road - Residency Road Cross",
            "Hosur Road": "Silk Board Junction",
        }
        target_junction = junction_names.get(location, f"{location} Central Intersection")
        barricade_directives = [
            f"Erect **{mitigated_barricades} physical barricades** at **{target_junction}** to control lane merges.",
            f"Deploy **VMS (Variable Message Signage)** at entry points warning motorists of **{event_type}** crowd congestion."
        ]

        # Signal adjustments
        signal_secs = 20 + signal_adjustment
        signal_directives = [
            f"Adjust signal timing: Inject **+{signal_secs} seconds green wave extension** on approaching corridors of **{location}**.",
            f"Synchronize signals at adjacent junctions to flush traffic from the source zone."
        ]

        # Special operations
        special_directives = []
        if crowd_size >= 15000:
            special_directives.append(f"Open **reversible center lane** on **{location}** for outbound traffic post-event.")
            special_directives.append("Position **2 heavy-duty recovery cranes** at corridor boundaries for immediate clearance.")
        else:
            special_directives.append(f"Open temporary lane bypasses for emergency vehicles on **{location}**.")

        if event_type == "VIP Movement":
            special_directives.append("Establish a 10-minute absolute rolling closure window with motorcycle escorts.")

        return {
            "officers_total": sum(allocations.values()) if allocations else mitigated_officers,
            "barricades_total": mitigated_barricades,
            "directives": {
                "patrols": patrol_directives,
                "diversions": diversion_directives,
                "barricades": barricade_directives,
                "signals": signal_directives,
                "special": special_directives
            }
        }
