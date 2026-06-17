"""
modules/report_generator.py
----------------------------
Generate downloadable reports (CSV, JSON, simple text/markdown) for active incident analysis.
"""

import csv
import io
import json

class ReportGenerator:
    @staticmethod
    def generate_incident_csv(forecast: dict, command_order: dict, allocations: dict) -> str:
        """
        Compiles active simulation data into a CSV string.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric Group", "Metric Name", "Value"])
        
        # Event Details
        writer.writerow(["Event", "Event Type", command_order.get("event_type", "N/A")])
        writer.writerow(["Event", "Location Corridor", command_order.get("location_corridor", "N/A")])
        writer.writerow(["Event", "Crowd Size", command_order.get("crowd_size", 0)])
        writer.writerow(["Event", "Weather", forecast.get("weather", "Sunny")])
        writer.writerow(["Event", "Lifecycle Stage", forecast.get("lifecycle", "During-event")])
        
        # Predictive Forecasts
        writer.writerow(["Forecast", "Expected Delay (minutes)", forecast.get("expected_delay", 0.0)])
        writer.writerow(["Forecast", "Congestion Risk (%)", forecast.get("congestion_risk", 0.0)])
        writer.writerow(["Forecast", "ML Confidence Score (%)", forecast.get("confidence", 0.0)])
        writer.writerow(["Forecast", "Vehicles Generated", forecast.get("vehicles_generated", 0)])
        
        # Dispatch & Planning
        writer.writerow(["Resources", "Officers Required", command_order.get("officers_total", 0)])
        writer.writerow(["Resources", "Active Diversions", command_order.get("diversions_count", 0)])
        writer.writerow(["Resources", "Signals green-wave adjust (seconds)", command_order.get("signals_wave_adjust", 0)])
        
        # Station-level allocations
        for zone, count in allocations.items():
            writer.writerow(["Allocation", f"Hebbal PS Corridor {zone}", count])
            
        # Economic Impact
        econ = forecast.get("economics", {})
        writer.writerow(["Economics", "Affected Traffic Flow (Vehicles)", econ.get("affected_vehicles", 0)])
        writer.writerow(["Economics", "Fuel Wasted (Liters)", econ.get("fuel_wasted_liters", 0.0)])
        writer.writerow(["Economics", "Economic Loss (INR Lakhs)", econ.get("economic_loss_lakhs", 0.0)])
        
        return output.getvalue()

    @staticmethod
    def generate_incident_json(forecast: dict, command_order: dict, allocations: dict) -> str:
        """
        Serializes active simulation parameters and outputs into structured JSON.
        """
        report_data = {
            "metadata": {
                "event_type": command_order.get("event_type"),
                "location_corridor": command_order.get("location_corridor"),
                "crowd_size": command_order.get("crowd_size"),
                "weather": forecast.get("weather"),
                "lifecycle": forecast.get("lifecycle")
            },
            "predictions": {
                "expected_delay_minutes": forecast.get("expected_delay"),
                "congestion_risk_pct": forecast.get("congestion_risk"),
                "ml_confidence_pct": forecast.get("confidence"),
                "vehicles_generated": forecast.get("vehicles_generated")
            },
            "economics": forecast.get("economics", {}),
            "command_directives": {
                "officers_deployed": command_order.get("officers_total"),
                "diversions_deployed": command_order.get("diversions_count"),
                "signal_wave_offset_sec": command_order.get("signals_wave_adjust"),
                "allocations": allocations,
                "directives_list": command_order.get("directives", {}).get("patrols", [])
            }
        }
        return json.dumps(report_data, indent=2)

    @staticmethod
    def generate_incident_markdown(forecast: dict, command_order: dict, allocations: dict) -> str:
        """
        Compiles an executive summary report in Markdown.
        """
        directives_str = "\n".join([f"- {d}" for d in command_order.get("directives", {}).get("patrols", [])])
        allocations_str = "\n".join([f"- **{z}**: {count} officers" for z, count in allocations.items()])
        econ = forecast.get("economics", {})
        
        md = f"""# TRAFFICLENS AI COMMANDER INCIDENT REPORT

## Executive Summary
An event of type **{command_order.get("event_type")}** has been analyzed at **{command_order.get("location_corridor")}** under **{forecast.get("weather")}** conditions.

* **Expected Delay:** {forecast.get("expected_delay")} minutes
* **Congestion Risk:** {forecast.get("congestion_risk")}%
* **ML Model Confidence:** {forecast.get("confidence")}%
* **Estimated Vehicles Generated:** {forecast.get("vehicles_generated")} vehicles

---

## Resource & Officer Deployments
The AI Traffic Commander has dispatched **{command_order.get("officers_total")} officers** with **{command_order.get("diversions_count")} diversions** and **{command_order.get("signals_wave_adjust")}s** green-wave offset overrides.

### Zone Allocation List:
{allocations_str}

### Tactical Action Directives:
{directives_str}

---

## Economic Damage Assessment
* **Affected Vehicles:** {econ.get("affected_vehicles")} vehicles
* **Excess Fuel Wasted:** {econ.get("fuel_wasted_liters")} Liters
* **Total Estimated Economic Loss:** ₹ {econ.get("economic_loss_lakhs")} Lakhs
"""
        return md.strip()

    @staticmethod
    def generate_html_report(forecast: dict, command_order: dict, allocations: dict) -> str:
        """
        Produces a self-contained, styled HTML report that can be opened in any browser
        and printed to PDF using Ctrl+P → Save as PDF (no wkhtmltopdf/weasyprint needed).
        """
        from datetime import datetime
        econ = forecast.get("economics", {})
        directives = command_order.get("directives", {}).get("patrols", [])
        dir_rows = "".join(f"<li>{d}</li>" for d in directives)
        alloc_rows = "".join(
            f"<tr><td>{z}</td><td style='text-align:center'>{c}</td></tr>"
            for z, c in allocations.items()
        )
        now = datetime.now().strftime("%d %b %Y, %I:%M %p")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TrafficLens AI — Incident Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; background: #f8fafc; color: #1e293b; padding: 40px; }}
  header {{ background: linear-gradient(135deg, #1e3a5f, #3b82f6); color: white;
            padding: 30px 40px; border-radius: 12px; margin-bottom: 30px; }}
  header h1 {{ font-size: 26px; font-weight: 700; }}
  header p {{ opacity: 0.85; margin-top: 6px; font-size: 14px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .kpi {{ background: white; border-radius: 10px; padding: 18px 20px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 4px solid #3b82f6; }}
  .kpi .label {{ font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
  .kpi .value {{ font-size: 28px; font-weight: 700; color: #1e3a5f; margin-top: 4px; }}
  .section {{ background: white; border-radius: 10px; padding: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 20px; }}
  .section h2 {{ font-size: 16px; font-weight: 700; color: #1e3a5f; border-bottom: 2px solid #e2e8f0;
                 padding-bottom: 10px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ background: #1e3a5f; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #e2e8f0; }}
  tr:hover {{ background: #f1f5f9; }}
  ul {{ padding-left: 20px; line-height: 1.8; font-size: 14px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px;
            font-weight: 600; background: #fef3c7; color: #92400e; }}
  .risk-high {{ background: #fee2e2; color: #991b1b; }}
  footer {{ text-align: center; color: #94a3b8; font-size: 12px; margin-top: 30px; }}
  @media print {{ body {{ background: white; padding: 20px; }} .section {{ box-shadow: none; border: 1px solid #e2e8f0; }} }}
</style>
</head>
<body>
<header>
  <h1>🚦 TrafficLens AI — Incident Command Report</h1>
  <p>Generated: {now} &nbsp;|&nbsp; Event: {command_order.get("event_type")} at {command_order.get("location_corridor")}</p>
</header>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Congestion Risk</div><div class="value">{forecast.get("congestion_risk")}%</div></div>
  <div class="kpi"><div class="label">Expected Delay</div><div class="value">{forecast.get("expected_delay")} min</div></div>
  <div class="kpi"><div class="label">ML Confidence</div><div class="value">{forecast.get("confidence")}%</div></div>
  <div class="kpi"><div class="label">Economic Loss</div><div class="value">₹{econ.get("economic_loss_lakhs")} L</div></div>
</div>

<div class="section">
  <h2>📋 Event Details</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Event Category</td><td>{command_order.get("event_type")}</td></tr>
    <tr><td>Location Corridor</td><td>{command_order.get("location_corridor")}</td></tr>
    <tr><td>Crowd Size</td><td>{command_order.get("crowd_size", 0):,} people</td></tr>
    <tr><td>Weather</td><td>{forecast.get("weather")}</td></tr>
    <tr><td>Lifecycle Phase</td><td>{forecast.get("lifecycle")}</td></tr>
    <tr><td>Vehicles Generated</td><td>{forecast.get("vehicles_generated", 0):,}</td></tr>
    <tr><td>Fuel Wasted</td><td>{econ.get("fuel_wasted_liters")} L</td></tr>
    <tr><td>Affected Vehicles</td><td>{econ.get("affected_vehicles", 0):,}</td></tr>
  </table>
</div>

<div class="section">
  <h2>👮 Officer Deployment Matrix</h2>
  <table>
    <tr><th>Zone / Corridor</th><th>Officers Deployed</th></tr>
    {alloc_rows}
  </table>
</div>

<div class="section">
  <h2>📢 Tactical Action Directives</h2>
  <ul>{dir_rows}</ul>
</div>

<footer>
  TrafficLens AI Commander &copy; {datetime.now().year} &nbsp;|&nbsp;
  For official use only. Print: Ctrl+P → Save as PDF.
</footer>
</body>
</html>"""
        return html

