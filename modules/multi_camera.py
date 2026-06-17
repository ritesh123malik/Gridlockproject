"""
modules/multi_camera.py
-------------------------
Multi-camera feed simulation for demo scalability.
Aggregates detections from different mock cameras located at key GPS coordinates.
"""

from modules.detection import MockDetector

class MultiCameraManager:
    def __init__(self):
        # Position cameras at coordinates corresponding to corridors:
        # Cam_1: Yelahanka/Hebbal area (Bellary Road)
        # Cam_2: CBD/MG Road area
        # Cam_3: Bellandur/ORR area
        self.cameras = {
            "Cam_1": {"lat": 13.0168, "lon": 77.5864, "detector": MockDetector(n_moving=4)},
            "Cam_2": {"lat": 12.9716, "lon": 77.5946, "detector": MockDetector(n_moving=3)},
            "Cam_3": {"lat": 12.9355, "lon": 77.6245, "detector": MockDetector(n_moving=5)},
        }

    def get_all_detections(self) -> dict:
        """
        Retrieves detections from each camera stream.
        Returns aggregated detections dictionary with coordinates.
        """
        results = {}
        for name, cfg in self.cameras.items():
            dets = cfg["detector"].detect()
            results[name] = {
                "detections": dets,
                "lat": cfg["lat"],
                "lon": cfg["lon"]
            }
        return results
