"""
detection.py
------------
Plug-in point for YOLOv8n. Drop your trained/finetuned weights in and
set USE_MOCK = False -- everything downstream (tracker, fusion,
dashboard) already works against this interface, so wiring in the real
model is a one-line change, not a rewrite.

Until the real weights are ready, MockDetector generates plausible
synthetic detections so the dashboard is demoable at every stage of
the build -- never show judges a blank screen because a model isn't
trained yet.
"""

from dataclasses import dataclass
import random


@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2) pixel coords
    vehicle_type: str    # car | bus | truck | lcv | two_wheeler | auto
    confidence: float


VEHICLE_CLASSES = ["car", "bus", "truck", "lcv", "two_wheeler", "auto"]

# Relative impact weight if a vehicle of this type is parked illegally --
# a bus blocking a lane disrupts flow far more than a parked two-wheeler.
VEHICLE_IMPACT_WEIGHT = {
    "bus": 1.0,
    "truck": 0.95,
    "lcv": 0.75,
    "car": 0.6,
    "auto": 0.4,
    "two_wheeler": 0.25,
}


class YOLOv8ParkingDetector:
    """
    Real detector. Lazy-imports ultralytics so this file can be imported
    even before `pip install ultralytics` / weights are available.

    Usage:
        det = YOLOv8ParkingDetector("weights/best.pt")
        boxes = det.detect(frame)   # frame: numpy BGR image (e.g. from cv2)
    """

    def __init__(self, weights_path: str, conf: float = 0.4):
        from ultralytics import YOLO  # noqa: deferred import
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame) -> list[Detection]:
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cls_name = self.model.names[int(box.cls[0])]
            out.append(Detection(
                bbox=(x1, y1, x2, y2),
                vehicle_type=cls_name,
                confidence=float(box.conf[0]),
            ))
        return out


class MockDetector:
    """
    Deterministic-ish synthetic detector for demos and UI development.
    Simulates a handful of vehicles that drift slightly frame-to-frame,
    plus one vehicle that stays put (the "illegal parking" case) so the
    dwell-time tracker has something real to accumulate against.
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720, n_moving: int = 3):
        self.frame_w, self.frame_h = frame_w, frame_h
        self._stationary_box = (
            random.randint(100, frame_w - 300),
            random.randint(100, frame_h - 200),
            0, 0,
        )
        x1, y1, _, _ = self._stationary_box
        self._stationary_box = (x1, y1, x1 + 120, y1 + 80)
        self._moving = []
        for _ in range(n_moving):
            x1 = random.randint(0, frame_w - 150)
            y1 = random.randint(0, frame_h - 100)
            self._moving.append([x1, y1, x1 + 110, y1 + 70])

    def detect(self, frame=None) -> list[Detection]:
        # The one vehicle that never moves -- this is the "illegal parking"
        # case the dwell-time tracker should flag once it crosses threshold.
        out = [Detection(
            bbox=self._stationary_box,
            vehicle_type=random.choice(["bus", "truck", "lcv"]),
            confidence=round(random.uniform(0.75, 0.95), 2),
        )]
        # Through-traffic: large per-step displacement keeps IoU with the
        # previous position low, so the tracker correctly treats these as
        # passing vehicles rather than letting them accumulate dwell time.
        for box in self._moving:
            box[0] = (box[0] + random.randint(40, 90)) % max(self.frame_w - 110, 1)
            box[1] += random.randint(-20, 20)
            box[1] = max(0, min(box[1], self.frame_h - 70))
            box[2] = box[0] + 110
            box[3] = box[1] + 70
            out.append(Detection(
                bbox=tuple(box),
                vehicle_type=random.choice(VEHICLE_CLASSES),
                confidence=round(random.uniform(0.6, 0.9), 2),
            ))
        return out
