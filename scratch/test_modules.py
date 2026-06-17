"""
scratch/test_modules.py
-----------------------
Tests all newly added modules including emergency corridors, adaptive signal timings,
spillover cascades, economic waste estimations, mode share, weather parameters,
event lifecycles, and police station resource constraints.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.forecasting import EventImpactForecaster
from modules.similarity import EventSimilaritySearch
from modules.rerouting import ReroutingEngine
from modules.optimization import PatrolOptimizer
from modules.learning import PostEventLearningSystem
from modules.commander import AITrafficCommander
from modules.emergency import EmergencyPriorityEngine
from modules.cascade import IncidentCascadePredictor
from modules.signal_opt import AdaptiveSignalOptimizer
from modules.alerts import CitizenAlertGenerator

def run_tests():
    print("🚦 [1/9] Testing Weather and Lifecycle Forecast Multipliers...")
    forecaster = EventImpactForecaster()
    
    # Use 2,000 crowd size to avoid saturation at the 99% cap
    normal_2k = forecaster.predict_impact("Concert", 2000, "ORR East 1", weather="Sunny", lifecycle="During-event")
    worse_2k = forecaster.predict_impact("Concert", 2000, "ORR East 1", weather="Heavy Rain", lifecycle="Exit-wave")
    
    assert worse_2k["expected_delay"] > normal_2k["expected_delay"], "Delay must increase in Rain + Exit-wave"
    assert worse_2k["congestion_risk"] > normal_2k["congestion_risk"], "Congestion risk must increase in Rain + Exit-wave"
    print("   ✅ Multipliers OK!")

    print("🚦 [2/9] Testing Transport Mode Share & Vehicles Generated...")
    # Concert: Cars=40%, Transit=30%, Bike=25%, Occupancies: Car=2, Transit=40, Bike=1.2
    # For 10,000 crowd:
    # Cars = 10000 * 0.4 / 2 = 2000
    # Transit = 10000 * 0.3 / 40 = 75
    # Bike = 10000 * 0.25 / 1.2 = 2083.33 -> Total ~ 4158 vehicles
    normal_10k = forecaster.predict_impact("Concert", 10000, "ORR East 1", weather="Sunny", lifecycle="During-event")
    assert abs(normal_10k["vehicles_generated"] - 4158) < 10, f"Expected ~4158 vehicles, got {normal_10k['vehicles_generated']}"
    print("   ✅ Mode Share OK!")

    print("🚦 [3/9] Testing Economic Impact Estimator...")
    assert normal_10k["economics"]["fuel_wasted_liters"] > 0, "Fuel wasted should be positive"
    assert normal_10k["economics"]["economic_loss_lakhs"] > 0.0, "Economic loss should be positive"
    print("   ✅ Economic Loss OK!")

    print("🚦 [4/9] Testing Explainable AI (XAI) Risk Breakdown...")
    xai = normal_10k["xai_breakdown"]
    total_xai = sum(xai.values())
    assert abs(total_xai - normal_10k["congestion_risk"]) < 1.0, f"XAI values sum ({total_xai}) must equal final risk ({normal_10k['congestion_risk']})"
    print("   ✅ XAI Attributions OK!")

    print("🚦 [5/9] Testing Constrained Officer Resource Allocation...")
    optimizer = PatrolOptimizer()
    affected_risks = {
        "Bellary Road 1": 90.0,  # Sadashivanagar PS capacity = 12
        "Bellary Road 2": 80.0,  # Yelahanka PS capacity = 10
        "ORR East 1": 70.0       # Bellandur PS capacity = 15
    }
    opt = optimizer.optimize_officers(affected_risks, 50)
    
    # Confirm allocations do not exceed station capacities
    for detail in opt["details"]:
        assert detail["officers"] <= detail["capacity_limit"], f"Allocation {detail['officers']} exceeds cap {detail['capacity_limit']} for {detail['zone']}"
    print("   ✅ Constrained Optimization OK!")

    print("🚦 [6/9] Testing Emergency Priority Green Corridor Engine...")
    routing = ReroutingEngine()
    emergency_eng = EmergencyPriorityEngine(routing)
    em_plan = emergency_eng.generate_green_corridor("Tumkur Road", "Mysore Road", "West of Chord Road")
    
    assert em_plan["status"] == "success", "Emergency routing failed"
    assert len(em_plan["green_corridor_path"]) > 1, "Green corridor path should exist"
    assert len(em_plan["signal_overrides"]) > 0, "Junction overrides should be generated"
    assert len(em_plan["lane_restrictions"]) > 0, "Vehicle restriction notices should exist"
    print("   ✅ Emergency Corridor Priority OK!")

    print("🚦 [7/9] Testing Incident Cascade spillover predictions...")
    cascade_pred = IncidentCascadePredictor()
    cascades = cascade_pred.predict_cascade("Bellary Road 1", 85.0, "Sunny")
    assert len(cascades) > 0, "Cascade prediction list should not be empty"
    assert cascades[0]["cascade_risk_pct"] > 0, "Spillover risks must be calculated"
    assert "status_level" in cascades[0], "Risk level string should exist"
    print("   ✅ Incident Cascade OK!")

    print("🚦 [8/9] Testing Adaptive Signal Timing Junction Offsets...")
    sig_opt = AdaptiveSignalOptimizer()
    plan = sig_opt.optimize_signal_timings("Bellary Road 1", ["Bellary Road 2", "ORR North 2"], 45.0)
    assert len(plan) > 0, "Signal plan cannot be empty"
    assert any(p["type"] == "Exit Flush" for p in plan), "Must contain exit flush action"
    assert any(p["type"] == "Entry Gating" for p in plan), "Must contain entry gating throttle action"
    print("   ✅ Adaptive Signals OK!")

    print("🚦 [9/9] Testing Citizen Alert SMS Copy...")
    alerts = CitizenAlertGenerator()
    alert = alerts.generate_citizen_alert("Concert", "ORR East 1", 35.0, ["ORR East 1", "CBD 1"], "Sunny")
    assert alert["sms_advisory"].startswith("TLENS ALERT:"), "SMS text copy must follow format"
    print("   ✅ Citizen Alerts OK!")

    print("\n🎉 ALL EXTENDED SYSTEM TESTS PASSED SUCCESSFULLY! 100% CORRECT.")

if __name__ == "__main__":
    run_tests()
