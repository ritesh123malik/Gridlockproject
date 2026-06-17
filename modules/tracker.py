"""
tracker.py
----------
Lightweight IoU-based multi-object tracker. No external deps beyond
numpy -- intentionally simple so it runs anywhere without GPU/heavy
install, and is easy to explain to judges.

Purpose: a YOLO model only tells you "there is a car here, in this
frame." It does NOT tell you how long that car has been sitting there.
Dwell time is what turns a detection into an illegal-parking *violation*
(a car briefly stopped at a signal is not a violation; a car stationary
for 8 minutes in a no-parking zone is) and it's the input that lets us
quantify impact rather than just detect presence.
"""

from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2) in pixel coords, last known
    vehicle_type: str
    lat: float
    lon: float
    first_seen: float    # unix timestamp
    last_seen: float
    plate: str = None
    misses: int = 0       # consecutive frames not matched
    prev_centroid: tuple = None   # (cx, cy) from previous frame
    speed_px_per_sec: float = 0.0 # estimated pixel speed

    @property
    def dwell_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def centroid(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def speed_kmh(self, pixels_per_meter: float = 8.0) -> float:
        """Converts pixel-displacement speed to km/h."""
        meters_per_sec = self.speed_px_per_sec / max(pixels_per_meter, 0.01)
        return round(meters_per_sec * 3.6, 1)


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class DwellTimeTracker:
    """
    Call `update(detections, lat, lon, now)` once per processed frame.
    `detections` is a list of dicts: {"bbox": (x1,y1,x2,y2), "vehicle_type": str}
    Returns the current list of TrackedObject, including dwell_seconds.

    Stationary-vehicle assumption: since the camera is fixed (typical
    traffic-pole CCTV), (lat, lon) is constant per camera and just gets
    attached to every detection from that feed.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 5):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: dict[int, TrackedObject] = {}
        self._next_id = 1

    def update(self, detections: list, lat: float, lon: float, now: float = None):
        now = now if now is not None else time.time()
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = list(range(len(detections)))
        matches = []  # (track_id, det_idx)

        # Greedy IoU matching (fine for the handful of objects in one frame)
        pairs = []
        for tid in unmatched_tracks:
            for di in unmatched_dets:
                iou = _iou(self._tracks[tid].bbox, detections[di]["bbox"])
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)
        used_tracks, used_dets = set(), set()
        for iou, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            matches.append((tid, di))

        # Update matched tracks
        for tid, di in matches:
            det = detections[di]
            t = self._tracks[tid]
            old_centroid = t.centroid
            t.prev_centroid = old_centroid
            t.bbox = det["bbox"]
            t.vehicle_type = det.get("vehicle_type", t.vehicle_type)
            dt = now - t.last_seen
            t.last_seen = now
            t.misses = 0
            if det.get("plate"):
                t.plate = det["plate"]
            # Compute pixel displacement speed
            new_cx, new_cy = t.centroid
            if t.prev_centroid and dt > 0:
                dx = new_cx - t.prev_centroid[0]
                dy = new_cy - t.prev_centroid[1]
                displacement = (dx**2 + dy**2) ** 0.5
                t.speed_px_per_sec = displacement / dt
            else:
                t.speed_px_per_sec = 0.0

        # New tracks for unmatched detections
        for di in unmatched_dets:
            if di in used_dets:
                continue
            det = detections[di]
            t = TrackedObject(
                track_id=self._next_id,
                bbox=det["bbox"],
                vehicle_type=det.get("vehicle_type", "unknown"),
                lat=lat, lon=lon,
                first_seen=now, last_seen=now,
                plate=det.get("plate"),
            )
            self._tracks[self._next_id] = t
            self._next_id += 1

        # Age out unmatched tracks
        for tid in list(unmatched_tracks):
            if tid not in used_tracks:
                self._tracks[tid].misses += 1
                if self._tracks[tid].misses > self.max_misses:
                    del self._tracks[tid]

        return list(self._tracks.values())

    def active_violations(self, min_dwell_seconds: float = 120.0):
        """Tracks stationary longer than the configured threshold."""
        return [t for t in self._tracks.values() if t.dwell_seconds >= min_dwell_seconds]

    def average_speed_kmh(self, pixels_per_meter: float = 8.0) -> float:
        """Computes mean speed of all moving vehicles (speed > 0) in km/h."""
        moving = [t for t in self._tracks.values() if t.speed_px_per_sec > 0.5]
        if not moving:
            return 0.0
        return round(sum(t.speed_kmh(pixels_per_meter) for t in moving) / len(moving), 1)
