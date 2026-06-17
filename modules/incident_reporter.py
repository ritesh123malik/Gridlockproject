"""
modules/incident_reporter.py
-----------------------------
Citizen crowdsourced incident reporting. Stores reports in data/citizen_reports.csv
and provides methods to load them for map overlay and priority fusion.
"""

import csv
import io
from datetime import datetime
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INCIDENT_TYPES = ["Congestion", "Accident", "Illegal Parking", "Road Damage", "Flooding", "Other"]
SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]

# Approximate corridor lat/lon lookup for auto-geocoding reports
CORRIDOR_COORDS = {
    "Bellary Road 1": (13.0168, 77.5864),
    "Bellary Road 2": (13.0481, 77.5880),
    "ORR East 1": (12.9355, 77.6245),
    "ORR East 2": (12.9150, 77.6785),
    "Mysore Road": (12.9524, 77.5248),
    "Hosur Road": (12.8927, 77.6385),
    "Bannerghatta Road": (12.8842, 77.5988),
    "Tumkur Road": (13.0112, 77.5244),
    "Magadi Road": (12.9620, 77.5084),
    "Old Madras Road": (13.0052, 77.6478),
    "CBD 2": (12.9716, 77.5946),
    "CBD 1": (12.9762, 77.6033),
    "West of Chord Road": (12.9862, 77.5444),
    "Old Airport Road": (12.9582, 77.6428),
    "Hennur Main Road": (13.0312, 77.6251),
    "Varthur Road": (12.9441, 77.7054),
    "Airport New South Road": (13.0680, 77.5980),
}


class CitizenIncidentReporter:
    def __init__(self):
        self.filepath = DATA_DIR / "citizen_reports.csv"
        self._init_db()

    def _init_db(self):
        if not self.filepath.exists():
            df = pd.DataFrame(columns=[
                "timestamp", "corridor", "incident_type", "severity",
                "description", "lat", "lon"
            ])
            df.to_csv(self.filepath, index=False)

    def submit_report(self, corridor: str, incident_type: str,
                      severity: str, description: str) -> dict:
        """Saves a citizen-submitted incident report."""
        coords = CORRIDOR_COORDS.get(corridor, (12.9716, 77.5946))
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "corridor": corridor,
            "incident_type": incident_type,
            "severity": severity,
            "description": description,
            "lat": coords[0],
            "lon": coords[1]
        }
        df = pd.read_csv(self.filepath)
        new_row_df = pd.DataFrame([row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(self.filepath, index=False)
        return row

    def get_reports(self) -> pd.DataFrame:
        """Returns all logged citizen reports."""
        return pd.read_csv(self.filepath)

    def get_report_count(self) -> int:
        return len(self.get_reports())

    def get_severity_color(self, severity: str) -> str:
        return {
            "Low": "#22c55e",
            "Medium": "#f59e0b",
            "High": "#ef4444",
            "Critical": "#7c3aed"
        }.get(severity, "#888888")
