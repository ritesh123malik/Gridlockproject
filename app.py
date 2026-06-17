"""
TrafficLens AI -- Streamlit Dashboard
======================================
AI-driven parking intelligence: detect illegal-parking hotspots and
quantify their impact on traffic flow to enable targeted enforcement.

Three tabs:
  1. Historical Intelligence -- real ASTRAM data, the congestion prior.
  2. Live Detection -- YOLOv8n plug-in point (mock detector until wired in).
  3. Fused Enforcement Dashboard -- live + historical -> priority ranking
     + patrol allocation. This is the answer to the problem statement.

Run with:  streamlit run app.py
"""

import time
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.detection import MockDetector, VEHICLE_IMPACT_WEIGHT
from modules.ocr import MockPlateReader
from modules.tracker import DwellTimeTracker
from modules.fusion import FusionEngine, allocate_patrols
from modules.historical import (
    load_corridor_scores, load_station_scores,
    load_historical_events, load_hourly_counts,
)
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

st.set_page_config(page_title="TrafficLens AI", layout="wide", page_icon="🚦")

# ---------------------------------------------------------------- state ---
for key, default in [
    ("tracker", None), ("fusion", None), ("mock_detector", None),
    ("sim_clock", None), ("frame_count", 0),
    ("forecaster", None), ("similarity", None), ("rerouting", None),
    ("optimizer", None), ("learning_sys", None), ("commander", None),
    ("emergency_engine", None), ("cascade_predictor", None),
    ("signal_optimizer", None), ("alert_generator", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.tracker is None:
    st.session_state.tracker = DwellTimeTracker()
if st.session_state.fusion is None:
    st.session_state.fusion = FusionEngine()
if st.session_state.sim_clock is None:
    st.session_state.sim_clock = time.time()
if st.session_state.forecaster is None:
    st.session_state.forecaster = EventImpactForecaster()
if st.session_state.similarity is None:
    st.session_state.similarity = EventSimilaritySearch()
if st.session_state.rerouting is None:
    st.session_state.rerouting = ReroutingEngine()
if st.session_state.optimizer is None:
    st.session_state.optimizer = PatrolOptimizer()
if st.session_state.learning_sys is None:
    st.session_state.learning_sys = PostEventLearningSystem()
if st.session_state.commander is None:
    st.session_state.commander = AITrafficCommander()
if st.session_state.emergency_engine is None:
    st.session_state.emergency_engine = EmergencyPriorityEngine(st.session_state.rerouting)
if st.session_state.cascade_predictor is None:
    st.session_state.cascade_predictor = IncidentCascadePredictor()
if st.session_state.signal_optimizer is None:
    st.session_state.signal_optimizer = AdaptiveSignalOptimizer()
if st.session_state.alert_generator is None:
    st.session_state.alert_generator = CitizenAlertGenerator()

st.title("🚦 TrafficLens AI")
st.caption(
    "AI-driven parking intelligence: detect illegal-parking hotspots and "
    "quantify their impact on traffic flow to enable targeted enforcement."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Historical Intelligence",
    "🎥 Live Detection",
    "🎯 Fused Enforcement Dashboard",
    "🔮 AI Traffic Commander (Forecast & Twin)",
    "🧠 Post-Event Learning System"
])

# =================================================== TAB 1: HISTORICAL ===
with tab1:
    st.subheader("Congestion Impact Intelligence — real ASTRAM data, Nov 2023–Apr 2024")
    st.info(
        "The source dataset has no labeled 'illegal parking' field. Scores below use "
        "vehicle-obstruction-type events (vehicle_breakdown, congestion, others, debris) "
        "as a physically valid proxy: something occupying the carriageway and disrupting "
        "flow — the same effect illegal parking has. This is documented in the submission writeup."
    )

    events = load_historical_events()
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.density_mapbox(
            events, lat="latitude", lon="longitude", z="is_obstruction",
            radius=8, center=dict(lat=12.97, lon=77.59), zoom=10,
            mapbox_style="open-street-map", height=520,
            color_continuous_scale="Inferno",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top corridors — Congestion Impact Score**")
        corridor_tab = load_corridor_scores()
        st.dataframe(
            corridor_tab.head(10)[["corridor", "total_events", "impact_score"]],
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Top police-station jurisdictions**")
        station_tab = load_station_scores()
        st.dataframe(
            station_tab.head(10)[["police_station", "total_events", "impact_score"]],
            hide_index=True, use_container_width=True,
        )

    st.markdown("**Hourly pattern of obstruction-type events (IST)**")
    hourly = load_hourly_counts()
    st.bar_chart(hourly.set_index("hour"))
    st.caption(
        "Peak hours skew toward night/early-morning, consistent with Bengaluru's "
        "daytime heavy-vehicle entry restrictions pushing truck/bus movement (and "
        "breakdowns) into night hours."
    )

# ================================================ TAB 2: LIVE DETECTION ===
with tab2:
    st.subheader("Live Detection Feed")
    st.caption(
        "YOLOv8n plug-in point. Runs on a mock detector by default so the pipeline is "
        "always demoable — swap in `YOLOv8ParkingDetector` from `modules/detection.py` "
        "once weights are trained; nothing downstream needs to change."
    )

    colA, colB = st.columns(2)
    with colA:
        cam_lat = st.number_input("Camera latitude", value=13.0168, format="%.6f")
        cam_lon = st.number_input("Camera longitude", value=77.5864, format="%.6f")
    with colB:
        dwell_threshold = st.slider("Illegal-parking dwell threshold (seconds)", 60, 900, 180, step=30)
        seconds_per_step = st.slider("Simulated seconds advanced per step (demo speed)", 5, 120, 60)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Advance simulated frame", use_container_width=True):
            if st.session_state.mock_detector is None:
                st.session_state.mock_detector = MockDetector()
            st.session_state.sim_clock += seconds_per_step
            dets = st.session_state.mock_detector.detect()
            dets_dicts = [{"bbox": d.bbox, "vehicle_type": d.vehicle_type} for d in dets]
            st.session_state.tracker.update(
                dets_dicts, cam_lat, cam_lon, now=st.session_state.sim_clock,
            )
            st.session_state.frame_count += 1
    with col2:
        if st.button("⟲ Reset feed", use_container_width=True):
            st.session_state.tracker = DwellTimeTracker()
            st.session_state.mock_detector = MockDetector()
            st.session_state.sim_clock = time.time()
            st.session_state.frame_count = 0
            st.rerun()

    tracked = list(st.session_state.tracker._tracks.values())
    if tracked:
        df = pd.DataFrame([{
            "track_id": t.track_id,
            "vehicle_type": t.vehicle_type,
            "dwell_seconds": round(t.dwell_seconds, 0),
            "status": "🚨 VIOLATION" if t.dwell_seconds >= dwell_threshold else "moving / transient",
        } for t in tracked]).sort_values("dwell_seconds", ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.write("No detections yet — click **Advance simulated frame** to start the feed.")

    n_violations = len(st.session_state.tracker.active_violations(dwell_threshold))
    st.metric("Active illegal-parking violations", n_violations)

    with st.expander("Wiring in your real model"):
        st.code(
            "from modules.detection import YOLOv8ParkingDetector\n"
            "detector = YOLOv8ParkingDetector('weights/best.pt')\n"
            "boxes = detector.detect(frame)  # frame: numpy BGR image from cv2\n",
            language="python",
        )
        st.write(
            "Frames from a real feed: `cv2.VideoCapture(source)`, read frames in a loop, "
            "call `detector.detect(frame)`, pass the result into "
            "`tracker.update(detections, cam_lat, cam_lon, now=frame_timestamp)`."
        )

# ============================================ TAB 3: FUSED ENFORCEMENT ===
with tab3:
    st.subheader("Fused Enforcement Priority")
    st.caption(
        "Combines live dwell-time detections with the historical Congestion Impact "
        "Score, time-of-day risk, and vehicle-type weight into a single explainable "
        "Enforcement Priority Score — directly answering the problem statement's "
        "'quantify impact to enable targeted enforcement.'"
    )

    violations = st.session_state.tracker.active_violations(min_dwell_seconds=dwell_threshold)
    ranked = st.session_state.fusion.rank(violations, now=datetime.now())

    if not ranked:
        st.write(
            "No active violations yet. Go to **Live Detection** and advance a few "
            "simulated frames until a vehicle's dwell time crosses the threshold."
        )
    else:
        rows = []
        for r in ranked:
            rows.append({
                "track_id": r.track_id,
                "corridor": r.corridor,
                "priority_score": r.score,
                "dwell_contribution": round(r.breakdown["dwell"][1] * 100, 1),
                "historical_contribution": round(r.breakdown["historical"][1] * 100, 1),
                "time_of_day_contribution": round(r.breakdown["time_of_day"][1] * 100, 1),
                "vehicle_type_contribution": round(r.breakdown["vehicle_type"][1] * 100, 1),
            })
        rank_df = pd.DataFrame(rows)
        st.markdown("**Ranked active violations**")
        st.dataframe(rank_df, hide_index=True, use_container_width=True)

        # Map: historical heatmap base layer + live violation markers on top
        st.markdown("**Map: historical hotspots (base) + live violations (markers)**")
        events = load_historical_events()
        fig = px.density_mapbox(
            events, lat="latitude", lon="longitude", z="is_obstruction",
            radius=8, center=dict(lat=12.97, lon=77.59), zoom=10,
            mapbox_style="open-street-map", height=480,
            color_continuous_scale="Inferno",
        )
        fig.add_trace(go.Scattermapbox(
            lat=[v.lat for v in violations], lon=[v.lon for v in violations],
            mode="markers",
            marker=dict(size=18, color="cyan", symbol="circle"),
            text=[f"track {v.track_id} · {v.vehicle_type} · {v.dwell_seconds:.0f}s" for v in violations],
            name="Live violations",
        ))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Patrol allocation
        st.markdown("**Patrol unit allocation**")
        zone_scores = {}
        for r in ranked:
            zone_scores[r.corridor] = zone_scores.get(r.corridor, 0) + r.score
        num_units = st.slider("Available patrol units", 1, 20, 5)
        allocation = allocate_patrols(zone_scores, num_units)
        st.dataframe(pd.DataFrame(allocation), hide_index=True, use_container_width=True)
        st.caption(
            "Largest-remainder proportional allocation: units distributed in proportion "
            "to each zone's aggregated priority score."
        )

# ============================================ TAB 4: EVENT FORECASTER & COMMANDER ===
with tab4:
    # 1. NASA-style Ops Room Header
    st.markdown(
        "<div style='background-color:#0E1117; padding:20px; border-radius:10px; border: 2px solid #3b82f6; text-align:center; margin-bottom:25px;'>"
        f"<h1 style='color:#3b82f6; margin:0; font-family:Outfit, Inter, sans-serif;'>🌌 NASA-STYLE TRAFFIC COMMAND OPS ROOM</h1>"
        f"<p style='color:#888; font-size:16px; margin:5px 0 0 0;'>Bengaluru Intelligent Traffic Command System | Multi-Agency Dispatch Center</p>"
        "</div>",
        unsafe_allow_html=True
    )

    col_setup, col_metrics = st.columns([1, 2])
    
    # 2. Controls Panel (Left Column)
    with col_setup:
        st.markdown("### 📋 Event Specifications")
        evt_type = st.selectbox(
            "Event Category",
            ["Political Rally", "Festival", "Sporting Event", "Concert", "VIP Movement"],
            key="evt_type"
        )
        corridors_list = sorted(list(st.session_state.forecaster.centroids["corridor"]))
        evt_loc = st.selectbox(
            "Incident Epicenter (Corridor)",
            corridors_list,
            index=corridors_list.index("Bellary Road 1") if "Bellary Road 1" in corridors_list else 0,
            key="evt_loc"
        )
        evt_crowd = st.slider(
            "Expected Crowd size",
            1000, 50000, 15000, step=1000,
            key="evt_crowd"
        )
        
        col_w, col_l = st.columns(2)
        with col_w:
            evt_weather = st.selectbox(
                "Base Weather",
                ["Sunny", "Light Rain", "Heavy Rain", "Fog"],
                key="evt_weather"
            )
        with col_l:
            evt_lifecycle = st.selectbox(
                "Lifecycle stage",
                ["Pre-event", "During-event", "Exit-wave", "Post-event"],
                index=1,
                key="evt_lifecycle"
            )

        st.markdown("### 🛠️ What-If Planning Simulator")
        sim_crowd_adj = st.slider(
            "Attendance Change (%)",
            -50, 100, 0, step=10,
            key="sim_crowd_adj"
        )
        sim_diversions = st.slider(
            "Deploy Diversions (Closed routes)",
            0, 3, 0, step=1,
            key="sim_diversions"
        )
        sim_signals = st.slider(
            "Optimize Signal Green Waves (seconds)",
            -10, 30, 0, step=5,
            key="sim_signals"
        )
        sim_officers = st.slider(
            "Available Officers pool",
            10, 100, 50, step=5,
            key="sim_officers"
        )

        st.markdown("### 🚑 Emergency priority Corridor Protection")
        active_emergency = st.checkbox("Enable Green Corridor Route Protection", value=False)
        
        nearby_corridors = [c for c in corridors_list if c != evt_loc]
        col_es, col_ee = st.columns(2)
        with col_es:
            emergency_start = st.selectbox(
                "Convoy Origin",
                nearby_corridors,
                index=0,
                disabled=not active_emergency
            )
        with col_ee:
            emergency_end = st.selectbox(
                "Convoy Destination (Hospital)",
                nearby_corridors,
                index=min(2, len(nearby_corridors)-1),
                disabled=not active_emergency
            )

    # 3. Time Playback Override & Computations
    st.markdown("---")
    playback_phase = st.select_slider(
        "🕒 SYSTEM TIMELINE PLAYBACK (Gridlock Evolution Simulation)",
        options=["Before Event (Pre-event)", "During Event (Peak)", "Exit Wave (Max Congestion)", "Recovery Phase"],
        value="During Event (Peak)"
    )

    # Map playback to active values
    if "Before Event" in playback_phase:
        active_crowd = int(evt_crowd * 0.25)
        active_weather = "Sunny"
        active_lifecycle = "Pre-event"
        active_diversions_val = 0
        active_signals_val = 0
        active_officers_val = max(10, int(sim_officers * 0.3))
    elif "During Event" in playback_phase:
        active_crowd = evt_crowd
        active_weather = evt_weather
        active_lifecycle = evt_lifecycle
        active_diversions_val = sim_diversions
        active_signals_val = sim_signals
        active_officers_val = sim_officers
    elif "Exit Wave" in playback_phase:
        active_crowd = int(evt_crowd * 1.25)
        active_weather = evt_weather
        active_lifecycle = "Exit-wave"
        active_diversions_val = sim_diversions
        active_signals_val = sim_signals
        active_officers_val = sim_officers
    else:  # Recovery
        active_crowd = int(evt_crowd * 0.35)
        active_weather = evt_weather
        active_lifecycle = "Post-event"
        active_diversions_val = sim_diversions
        active_signals_val = sim_signals
        active_officers_val = sim_officers

    correction = st.session_state.learning_sys.get_correction_factor()
    
    # Run Baseline
    base_results = st.session_state.forecaster.predict_impact(
        evt_type, evt_crowd, evt_loc, evt_weather, evt_lifecycle, correction_factor=correction
    )
    
    # Run Simulator (What-If with active playback overrides)
    simulated_crowd = int(active_crowd * (1.0 + sim_crowd_adj / 100.0))
    sim_results = st.session_state.forecaster.predict_impact(
        evt_type, simulated_crowd, evt_loc, active_weather, active_lifecycle, correction_factor=correction
    )
    
    # Apply what-if mitigations
    if active_diversions_val > 0:
        sim_results["expected_delay"] = max(2.0, sim_results["expected_delay"] * (1.0 - 0.15 * active_diversions_val))
        sim_results["congestion_risk"] = max(10.0, sim_results["congestion_risk"] * (1.0 - 0.12 * active_diversions_val))
        
    if active_signals_val > 0:
        sim_results["expected_delay"] = max(2.0, sim_results["expected_delay"] * (1.0 - 0.005 * active_signals_val))
        sim_results["congestion_risk"] = max(10.0, sim_results["congestion_risk"] * (1.0 - 0.003 * active_signals_val))

    # Constrained Officer Optimization
    affected_risks = {x["corridor"]: sim_results["congestion_risk"] * (1.0 - 0.08 * idx) for idx, x in enumerate(sim_results["affected_corridors"][:6])}
    opt_results = st.session_state.optimizer.optimize_officers(affected_risks, active_officers_val)
    
    mitigated_risk = opt_results["network_risk_after"]
    risk_reduction_ratio = mitigated_risk / max(1.0, opt_results["network_risk_before"])
    sim_results["expected_delay"] = max(2.0, round(sim_results["expected_delay"] * (0.55 + 0.45 * risk_reduction_ratio), 1))
    sim_results["congestion_risk"] = max(10.0, round(mitigated_risk, 1))

    # Detours Calculation
    nearby = [x["corridor"] for x in base_results["affected_corridors"] if x["corridor"] != evt_loc]
    start_route = nearby[0] if len(nearby) > 0 else evt_loc
    end_route = nearby[1] if len(nearby) > 1 else corridors_list[-1]
    detours = st.session_state.rerouting.calculate_detours(start_route, end_route, evt_loc)
    detour_a_nodes = detours["route_a"]["nodes"] if detours.get("status") == "success" else []

    # Junction Signal Timing Offsets
    signal_plan = st.session_state.signal_optimizer.optimize_signal_timings(
        evt_loc, [x["corridor"] for x in sim_results["affected_corridors"]], sim_results["expected_delay"]
    )

    # Cascading spillover predictions
    cascades = st.session_state.cascade_predictor.predict_cascade(evt_loc, sim_results["congestion_risk"], active_weather)

    # Emergency Priority Corridor Routing
    emergency_plan = None
    if active_emergency:
        emergency_plan = st.session_state.emergency_engine.generate_green_corridor(
            emergency_start, emergency_end, evt_loc
        )

    # Citizen push advisories
    citizen_alert = st.session_state.alert_generator.generate_citizen_alert(
        evt_type, evt_loc, sim_results["expected_delay"], detour_a_nodes, active_weather
    )

    # Tactical Command decisions
    command_order = st.session_state.commander.generate_command_order(
        evt_type, evt_loc, simulated_crowd, sim_results, opt_results["allocations"], detours, active_diversions_val, active_signals_val
    )

    # 4. NASA City Health Score & Metrics (Right Column)
    with col_metrics:
        city_health_score = int(100 - sim_results["congestion_risk"] * 0.45)
        st.markdown(
            f"<div style='background-color:#0E1117; padding:15px; border-radius:10px; border: 1px solid #ff4b4b; text-align:center; margin-bottom:15px;'>"
            f"<h2 style='color:#ff4b4b; margin:0; font-family:Outfit;'>🚨 LIVE CITY HEALTH SCORE: {city_health_score}/100</h2>"
            f"<p style='color:#888; margin:5px 0 0 0;'>System State: <b>{'STABLE FLOW' if city_health_score > 80 else ('CONGESTED' if city_health_score > 55 else 'CRITICAL GRIDLOCK')}</b></p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("### 📊 ML Predictive Output")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Congestion Risk", f"{sim_results['congestion_risk']:.1f}%")
        m_col2.metric("Expected Delay", f"{sim_results['expected_delay']:.1f} min")
        m_col3.metric("ML Model Trust Score", f"{sim_results['confidence']:.1f}%")
        m_col4.metric("Officers Required", opt_results["required_officers"])

        # Explainable AI Horizontal Bar Chart
        st.markdown("**Explainable AI (XAI) - Forecast Risk Point Attribution**")
        xai_data = pd.DataFrame({
            "Risk Contribution Factor": list(sim_results["xai_breakdown"].keys()),
            "Risk Score Points": list(sim_results["xai_breakdown"].values())
        })
        fig_xai = px.bar(
            xai_data, y="Risk Contribution Factor", x="Risk Score Points",
            orientation="h", text="Risk Score Points",
            color="Risk Score Points", color_continuous_scale="Reds",
            height=200
        )
        fig_xai.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_xai, use_container_width=True)

    # 5. economic Counter (ROI) and AI vs Traditional Planning Comparison
    st.markdown("---")
    st.markdown("### 💳 ROI Economic Impact & Strategy Comparison")
    
    # Cost math
    traditional_delay = round(base_results["expected_delay"] * 1.5, 1)
    traditional_cost = round(base_results["economics"]["economic_loss_lakhs"] * 1.7, 2)
    traditional_officers = int(base_results["resource_need"]["officers"] * 1.3)
    
    ai_delay = sim_results["expected_delay"]
    ai_cost = sim_results["economics"]["economic_loss_lakhs"]
    ai_officers = command_order["officers_total"]
    
    savings_lakhs = round(traditional_cost - ai_cost, 2)
    
    roi_col1, roi_col2, roi_col3 = st.columns(3)
    roi_col1.metric("Traditional Plan Cost", f"₹ {traditional_cost:.2f} Lakh", delta=None)
    roi_col2.metric("AI Commander Plan Cost", f"₹ {ai_cost:.2f} Lakh", delta=f"- {savings_lakhs:.2f} Lakh", delta_color="inverse")
    roi_col3.metric("Total Net Savings", f"₹ {savings_lakhs:.2f} Lakh", delta="45% Cost Reduction")

    # Side-by-Side Comparison
    compare_planning_df = pd.DataFrame({
        "Operational Metric": ["Expected Delay (min)", "Economic Loss (₹ Lakh)", "Officers Deployed"],
        "Traditional Dispatch": [f"{traditional_delay} min", f"₹ {traditional_cost:.2f} Lakh", f"{traditional_officers} officers"],
        "AI Traffic Commander": [f"{ai_delay} min", f"₹ {ai_cost:.2f} Lakh", f"{ai_officers} officers"]
    })
    st.dataframe(compare_planning_df, hide_index=True, use_container_width=True)

    # Deficit Warnings & Borrow Plan
    if opt_results["deficit"] > 0:
        st.error(
            f"⚠️ **RESOURCE DEFICIT DETECTED:** Required **{opt_results['required_officers']}** officers, "
            f"but only **{active_officers_val}** are available. Shortfall of **{opt_results['deficit']}** officers."
        )
        borrow_texts = []
        for borrow in opt_results["borrowing_plan"]:
            borrow_texts.append(f"Borrow **{borrow['borrow_count']} officers** from **{borrow['station']}** (Spare capacity)")
        st.info("💡 **AI Emergency Borrowing Plan:**\n" + "\n".join([f"- {t}" for t in borrow_texts]))

    # 6. Incident Cascade Chain
    st.markdown("### 🔗 Incident Cascade Propagation Chain")
    chain_html = f"<div style='background-color:#1e1e1e; padding:15px; border-radius:5px; font-family:Courier New; font-size:16px; color:#ffcc00; text-align:center;'>"
    chain_html += f"<b>Event Source:</b> {evt_loc}"
    
    # Get top 3 cascade corridors
    for idx, c in enumerate(cascades[:3]):
        chain_html += f" ➔ <b>{c['corridor']}</b> ({c['cascade_risk_pct']}% Risk | gridlock in {c['time_to_gridlock_mins']}m)"
    chain_html += "</div>"
    st.markdown(chain_html, unsafe_allow_html=True)

    st.markdown("---")

    # 7. Stakeholder Switcher Panel
    st.markdown("### 👤 Select Stakeholder Dispatch Console")
    stakeholder = st.radio(
        "Viewport",
        ["👮 Traffic Police Ops", "🛣️ Traffic Engineer", "🚑 Emergency Services", "👥 Citizen Information Feed"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Render specific content based on active stakeholder
    if stakeholder == "👮 Traffic Police Ops":
        st.markdown("### 👮 Police Dispatch Console")
        st.info("Direct patrol allocations, violation tracking, and inter-precinct borrowing directives.")
        
        pol_col1, pol_col2 = st.columns([1, 1])
        with pol_col1:
            st.markdown("#### 🚨 Officer Deployment Directives")
            for p in command_order["directives"]["patrols"]:
                st.markdown(f"- {p}")
        with pol_col2:
            st.markdown("#### 👮 Precinct Capacity Allocation Matrix")
            st.dataframe(
                pd.DataFrame(opt_results["details"])[["zone", "station", "base_risk", "final_risk", "officers", "capacity_limit", "risk_reduction_pct"]],
                hide_index=True, use_container_width=True
            )

    elif stakeholder == "🛣️ Traffic Engineer":
        st.markdown("### 🛣️ Traffic Engineering Control Room")
        st.info("Configure adaptive offsets, track cascade spillovers, and monitor arterial sensor health.")
        
        eng_col1, eng_col2 = st.columns([1, 1])
        with eng_col1:
            st.markdown("#### 🚦 Adaptive Signal Control Plan")
            sig_df = pd.DataFrame(signal_plan)
            st.dataframe(
                sig_df[["junction", "type", "offset_seconds", "recommended_action"]],
                hide_index=True, use_container_width=True
            )
        with eng_col2:
            st.markdown("#### ⚠️ Spillover Cascade Warnings")
            cas_df = pd.DataFrame(cascades)
            if not cas_df.empty:
                st.dataframe(
                    cas_df[["corridor", "distance_km", "cascade_risk_pct", "queue_length_meters", "time_to_gridlock_mins", "status_level"]],
                    hide_index=True, use_container_width=True
                )
            else:
                st.write("No cascades predicted.")

    elif stakeholder == "🚑 Emergency Services":
        st.markdown("### 🚑 Emergency Services Center")
        st.info("Monitor Green Corridor routes and bypass congestion blockages in real-time.")
        
        if active_emergency and emergency_plan:
            st.success(
                f"Convoy Medical Bypass Locked: **{emergency_start}** to **{emergency_end}** "
                f"({emergency_plan['distance_km']} km | {emergency_plan['travel_time_min']} min transit)"
            )
            for override in emergency_plan["signal_overrides"]:
                st.markdown(f"- {override}")
            for rest in emergency_plan["lane_restrictions"]:
                st.markdown(f"- {rest}")
        else:
            st.warning("⚠️ Green Corridor Protection is currently disabled. Toggle the checkbox in the specifications panel to compute priority routes.")

    else:  # Public View
        st.markdown("### 👥 Public Citizen Advisory Board")
        st.info("Broadcast feeds warning motorists of delays, alternate detours, and transit options.")
        
        pub_col1, pub_col2 = st.columns([1, 1])
        with pub_col1:
            st.markdown("#### 📢 Advisory Advisory")
            with st.container(border=True):
                st.markdown(f"### {citizen_alert['title']}")
                st.markdown(f"**⏰ Time:** {citizen_alert['time_window']}")
                st.markdown(citizen_alert["delay_warning"])
                if citizen_alert["weather_warning"]:
                    st.markdown(citizen_alert["weather_warning"])
                st.markdown(citizen_alert["detour_path"])
                st.markdown(citizen_alert["transit_recommendation"])
        with pub_col2:
            st.markdown("#### 💬 SMS Broadcast Feed copy")
            st.code(citizen_alert["sms_advisory"])
            
            st.markdown("#### 🗺️ Alternative Detours")
            if detours["status"] == "success":
                detour_rows = []
                for name, rkey in [("Route A (Detour)", "route_a"), ("Route B (Secondary)", "route_b")]:
                    r = detours[rkey]
                    detour_rows.append({
                        "Option": name,
                        "Detour Path": " → ".join(r["nodes"]),
                        "Distance": f"{r['distance_km']} km",
                        "Est. Time": f"{r['travel_time_min']} min"
                    })
                st.dataframe(pd.DataFrame(detour_rows), hide_index=True, use_container_width=True)

    # 8. Digital Twin Map View
    st.markdown("---")
    st.markdown("### 🗺️ Digital Twin City Map - Traffic Propagation Simulator")
    st.caption("Visualizes how traffic congests adjacent corridors over time and how diversion directives clear it.")

    map_placeholder = st.empty()
    sim_status = st.empty()
    sim_btn = st.button("▶️ Run Propagation Simulation Animation", use_container_width=True)
    
    def draw_twin_map(current_state_colors: dict, route_nodes: list = None, emer_nodes: list = None):
        centroids_df = st.session_state.forecaster.centroids.copy()
        c_lat, c_lon = st.session_state.rerouting.get_corridor_coordinates(evt_loc)
        fig_twin = go.Figure()
        
        # Add corridors as nodes
        lats, lons, colors, sizes, texts = [], [], [], [], []
        for _, row in centroids_df.iterrows():
            name = row["corridor"]
            status = current_state_colors.get(name, "GREEN")
            lats.append(row["centroid_lat"])
            lons.append(row["centroid_lon"])
            
            if status == "RED":
                colors.append("red")
                sizes.append(20)
            elif status == "YELLOW":
                colors.append("orange")
                sizes.append(15)
            else:
                colors.append("green")
                sizes.append(10)
            
            texts.append(f"{name} · Status: {status}")
            
        fig_twin.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="markers",
            marker=dict(size=sizes, color=colors),
            text=texts,
            name="Corridors"
        ))
        
        # Draw detour overlay
        if route_nodes:
            route_lats = [st.session_state.rerouting.get_corridor_coordinates(n)[0] for n in route_nodes]
            route_lons = [st.session_state.rerouting.get_corridor_coordinates(n)[1] for n in route_nodes]
            fig_twin.add_trace(go.Scattermapbox(
                lat=route_lats, lon=route_lons,
                mode="lines+markers",
                line=dict(width=4, color="cyan"),
                marker=dict(size=8, color="cyan"),
                name="Detour Route A"
            ))

        # Draw emergency green corridor overlay
        if emer_nodes:
            emer_lats = [st.session_state.rerouting.get_corridor_coordinates(n)[0] for n in emer_nodes]
            emer_lons = [st.session_state.rerouting.get_corridor_coordinates(n)[1] for n in emer_nodes]
            fig_twin.add_trace(go.Scattermapbox(
                lat=emer_lats, lon=emer_lons,
                mode="lines+markers",
                line=dict(width=5, color="magenta"),
                marker=dict(size=10, color="magenta"),
                name="🚑 Green Corridor Priority"
            ))

        # Highlight Event Origin
        fig_twin.add_trace(go.Scattermapbox(
            lat=[c_lat], lon=[c_lon],
            mode="markers+text",
            marker=dict(size=25, color="purple", symbol="circle"),
            text=[f"★ EVENT ORIGIN: {evt_loc}"],
            textposition="top center",
            name="Event Origin"
        ))

        fig_twin.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=12.975, lon=77.60), zoom=10.5),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)")
        )
        return fig_twin

    active_emer_path = emergency_plan["green_corridor_path"] if (active_emergency and emergency_plan) else None

    if sim_btn:
        centroids_df = st.session_state.forecaster.centroids.copy()
        all_corridors = list(centroids_df["corridor"])
        neighbors = [x["corridor"] for x in base_results["affected_corridors"] if x["corridor"] != evt_loc]
        n_close = neighbors[:2]
        n_far = neighbors[2:5]
        
        steps = [
            ("Step 1: Event Commences - local congestion builds up at origin.", 
             {evt_loc: "RED"}, None, None),
            ("Step 2: Spillover Begins - traffic backs up onto adjacent corridors.", 
             {evt_loc: "RED", **{n: "YELLOW" for n in n_close}}, None, None),
            ("Step 3: Peak Saturation - adjacent corridors saturate, outer grid slows down.", 
             {evt_loc: "RED", **{n: "RED" for n in n_close}, **{n: "YELLOW" for n in n_far}}, None, None),
            ("Step 4: AI Commander Activates Detour Routes & Overrides Signals.", 
             {evt_loc: "RED", **{n: "YELLOW" for n in n_close}, **{n: "GREEN" for n in n_far}}, detour_a_nodes, active_emer_path),
            ("Step 5: Mitigation Effective - grid returns to stable flow, emergency routes locked clear.", 
             {evt_loc: "YELLOW", **{n: "GREEN" for n in n_close}}, detour_a_nodes, active_emer_path)
        ]
        
        for idx, (desc, color_map, r_nodes, em_nodes) in enumerate(steps):
            sim_status.info(f"⚡ **Animation Step {idx+1}/5**: {desc}")
            full_color_map = {c: "GREEN" for c in all_corridors}
            full_color_map.update(color_map)
            map_placeholder.plotly_chart(draw_twin_map(full_color_map, r_nodes, em_nodes), use_container_width=True)
            time.sleep(1.2)
            
        sim_status.success("🎉 Simulation Animation Completed! Traffic mitigated successfully via Commander Directives.")
    else:
        initial_colors = {c: "GREEN" for c in st.session_state.forecaster.centroids["corridor"]}
        initial_colors[evt_loc] = "RED"
        map_placeholder.plotly_chart(draw_twin_map(initial_colors, None, active_emer_path), use_container_width=True)
        sim_status.caption("Click the button above to animate traffic propagation and routing response.")

    # Similarity search display
    st.markdown("### 🔍 Most Similar Historical Events (Decision-Support)")
    st.caption("Matches current parameters to past major events to pull actions and lessons learned.")
    sim_events = st.session_state.similarity.find_similar_events(evt_type, simulated_crowd, evt_loc)
    
    if sim_events:
        for idx, se in enumerate(sim_events):
            with st.expander(f"Match #{idx+1}: {se['event_type']} at {se['location']} ({se['similarity_pct']}% Match)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Crowd Size:** {se['crowd_size']} people")
                    st.markdown(f"**Actual Delay:** {se['actual_delay']} min | **Actual Risk:** {se['actual_risk']}%")
                    st.markdown(f"**Distance to Event:** {se['distance_km']} km")
                with col2:
                    st.markdown(f"**Actions Taken:** {se['actions_taken']}")
                    st.markdown(f"**Lessons Learned:** {se['lessons_learned']}")
    else:
        st.write("No historical events matched the criteria.")

# ============================================ TAB 5: POST-EVENT LEARNING =============
with tab5:
    st.subheader("🧠 Post-Event Learning System (Feedback Loop)")
    st.caption(
        "Closes the loop on the ASTRAM problem statement by tracking predictions against "
        "actual outcomes, calculating system bias, and automatically calibrating forecasting weights."
    )

    col_hist, col_feedback = st.columns([2, 1])

    with col_hist:
        st.markdown("### 📊 Learning Logs & Calibration Model")
        history_df = st.session_state.learning_sys.get_history()
        
        total_runs = len(history_df)
        avg_err = history_df["difference"].mean()
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Events Logged", total_runs)
        metric_col2.metric("Mean Forecasting Bias", f"{avg_err:+.1f} min")
        metric_col3.metric("Calibrated Correction Factor", f"{correction*100:+.1f}%")

        st.dataframe(history_df.sort_index(ascending=False), hide_index=True, use_container_width=True)
        
        st.markdown("**Predicted vs. Actual Congestion Delays**")
        fig_scatter = px.scatter(
            history_df, x="predicted_delay", y="actual_delay",
            labels={"predicted_delay": "Predicted Delay (min)", "actual_delay": "Actual Delay (min)"},
            hover_data=["event_type", "location", "crowd_size"],
            trendline="ols",
            height=300
        )
        max_val = max(history_df["predicted_delay"].max(), history_df["actual_delay"].max())
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash", color="red"),
            x0=0, y0=0, x1=max_val, y1=max_val
        )
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_feedback:
        st.markdown("### 📝 Log Post-Event Outcome")
        st.info("Input real-world outcomes of a recent event to update the model calibration weights.")
        
        with st.form("feedback_form"):
            fb_type = st.selectbox("Event Type", ["Political Rally", "Festival", "Sporting Event", "Concert", "VIP Movement"])
            fb_loc = st.selectbox("Location (Corridor)", corridors_list)
            fb_crowd = st.number_input("Crowd Size", min_value=1000, max_value=100000, value=15000, step=1000)
            
            fb_pred = st.session_state.forecaster.predict_impact(fb_type, fb_crowd, fb_loc, correction_factor=correction)
            
            st.markdown(f"**Forecasted Delay:** {fb_pred['expected_delay']} min")
            st.markdown(f"**Forecasted Risk:** {fb_pred['congestion_risk']}%")
            
            fb_actual_delay = st.slider("Actual Measured Delay (min)", 2, 90, int(fb_pred['expected_delay']))
            fb_actual_risk = st.slider("Actual Measured Risk (%)", 10, 100, int(fb_pred['congestion_risk']))
            
            submit_fb = st.form_submit_button("💾 Save Outcome & Train Model")
            
            if submit_fb:
                new_row = st.session_state.learning_sys.log_event_feedback(
                    fb_type, fb_loc, fb_crowd,
                    fb_pred['expected_delay'], fb_actual_delay,
                    fb_pred['congestion_risk'], fb_actual_risk
                )
                st.success(
                    f"Outcome logged! Prediction error of **{new_row['difference']:+.1f} min** "
                    f"incorporated into forecasting calibration."
                )
                st.rerun()

        if st.button("⟲ Reset Learning System & Seed Data", use_container_width=True):
            st.session_state.learning_sys.reset_learning()
            st.success("Learning system database reset successfully.")
            st.rerun()
