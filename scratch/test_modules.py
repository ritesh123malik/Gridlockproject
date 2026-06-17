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

    print("🚦 [10/12] Testing Advanced Core Modules (Camera, Reports, Routes, Chat)...")
    from modules.multi_camera import MultiCameraManager
    from modules.report_generator import ReportGenerator
    from modules.route_optimizer import RouteOptimizer
    from modules.chat_assistant import TrafficChatAssistant
    
    # 1. Multi-camera
    mc = MultiCameraManager()
    dets = mc.get_all_detections()
    assert "Cam_1" in dets and "Cam_2" in dets and "Cam_3" in dets
    assert len(dets["Cam_1"]["detections"]) > 0
    
    # 2. Reports
    f_mock = {"expected_delay": 25.0, "congestion_risk": 75.0, "confidence": 92.0, "vehicles_generated": 1200, "weather": "Sunny", "lifecycle": "During-event", "economics": {"affected_vehicles": 3000, "fuel_wasted_liters": 250.0, "economic_loss_lakhs": 3.50}}
    cmd_mock = {"event_type": "Concert", "location_corridor": "Bellary Road 1", "crowd_size": 15000, "officers_total": 45, "diversions_count": 1, "signals_wave_adjust": 10, "directives": {"patrols": ["Patrol 1", "Patrol 2"]}}
    alloc_mock = {"Bellary Road 1": 15, "ORR East 1": 10}
    
    csv_str = ReportGenerator.generate_incident_csv(f_mock, cmd_mock, alloc_mock)
    assert "Expected Delay" in csv_str and "Concert" in csv_str
    
    json_str = ReportGenerator.generate_incident_json(f_mock, cmd_mock, alloc_mock)
    assert '"expected_delay_minutes": 25.0' in json_str
    
    md_str = ReportGenerator.generate_incident_markdown(f_mock, cmd_mock, alloc_mock)
    assert "# TRAFFICLENS" in md_str
    
    # 3. Route Optimizer
    ro = RouteOptimizer(routing)
    r_normal = ro.get_optimal_route("Tumkur Road", "Mysore Road")
    assert r_normal["status"] == "success"
    r_avoid = ro.get_optimal_route("Tumkur Road", "Mysore Road", avoid_corridors=["West of Chord Road"])
    assert r_avoid["status"] == "success"
    
    # 4. Chat Assistant
    chat = TrafficChatAssistant(f_mock, {"route_a": {"nodes": ["Node1", "Node2"], "distance_km": 5.0, "travel_time_min": 10.0}}, cmd_mock)
    ans_delay = chat.respond("What is the delay?")
    assert "25.0 minutes" in ans_delay
    ans_route = chat.respond("Can you give me detours?")
    assert "Node1 ➔ Node2" in ans_route
    print("   ✅ Core Modules OK!")

    print("🚦 [11/12] Testing Social Sentiment & ROI Cost-Benefit Analysis...")
    from modules.sentiment import SentimentAnalyzer
    from modules.cost_benefit import CostBenefitAnalyzer
    
    # 1. Sentiment
    sa = SentimentAnalyzer()
    s_sunny = sa.analyze("Concert", 10000, "Sunny")
    s_rain = sa.analyze("Concert", 10000, "Heavy Rain")
    assert s_sunny["sentiment_score"] > s_rain["sentiment_score"], "Rain must penalize sentiment score"
    
    # 2. ROI
    roi = CostBenefitAnalyzer.compute_roi(2.5, 20.0) 
    assert roi == 550.0, f"Expected 550.0%, got {roi}"
    
    proj = CostBenefitAnalyzer.get_savings_projections(2.5, 20.0)
    assert proj["payback_months"] == 1.8, f"Expected 1.8, got {proj['payback_months']}"
    print("   ✅ Sentiment & Cost-Benefit OK!")

    print("🚦 [12/12] Testing Model Auto-Retraining & Reinforcement Learning Signal choice...")
    from modules.model_retrainer import ModelRetrainer
    from modules.signal_controller import AdaptiveSignalController
    from modules.city_correlator import CityCorrelator
    import pandas as pd
    
    # 1. Retrainer
    fb_df = pd.DataFrame([{"location": "ORR East 1", "event_type": "Concert", "crowd_size": 1000, "actual_delay": 20.0, "actual_risk": 50.0}] * 3)
    retrainer = ModelRetrainer(forecaster.ml_predictor, threshold=5)
    assert not retrainer.should_retrain(fb_df)
    fb_df_long = pd.DataFrame([{"location": "ORR East 1", "event_type": "Concert", "crowd_size": 1000, "actual_delay": 20.0, "actual_risk": 50.0}] * 6)
    assert retrainer.should_retrain(fb_df_long)
    
    # Verify retraining execution
    retrained = retrainer.retrain()
    assert retrained is True
    
    # 2. Signal Controller
    sc = AdaptiveSignalController()
    action = sc.get_action("HIGH_CONGESTION")
    assert action in [0, 1, 2]
    
    # 3. City Correlator
    cc = CityCorrelator()
    corr_results = cc.correlate("Concert")
    assert len(corr_results) > 0
    assert any(c["city"] == "London" for c in corr_results)
    print("   ✅ Retraining, RL & City Correlator OK!")

    print("\n🎉 ALL EXTENDED SYSTEM TESTS PASSED SUCCESSFULLY! 100% CORRECT.")

if __name__ == "__main__":
    run_tests()
