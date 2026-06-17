# TrafficLens AI

AI-driven parking intelligence: detect illegal-parking hotspots and quantify
their impact on traffic flow to enable targeted enforcement.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the app, go to the **Live Detection** tab, click **Advance simulated
frame** ~4-5 times, then check the **Fused Enforcement Dashboard** tab to
see the priority ranking and patrol allocation populate.

## Architecture

```
                ┌─────────────────────┐
 Camera feed →  │  detection.py        │  YOLOv8n (plug in your weights;
 (video/cv2)    │  YOLOv8ParkingDet.   │  MockDetector until then)
                └──────────┬───────────┘
                           │ bboxes + vehicle_type
                           ▼
                ┌─────────────────────┐
                │  tracker.py          │  IoU matching across frames →
                │  DwellTimeTracker    │  dwell_seconds per tracked vehicle
                └──────────┬───────────┘
                           │ TrackedObject (lat, lon, dwell_seconds, ...)
                           ▼
                ┌─────────────────────┐      ┌──────────────────────────┐
                │  fusion.py           │ ←──  │  historical.py            │
                │  FusionEngine.score  │      │  CorridorMatcher,         │
                │  = dwell + history   │      │  hourly risk, scores      │
                │  + time + vehicle    │      │  (built from real ASTRAM  │
                └──────────┬───────────┘      │  event data)              │
                           │ PriorityResult    └──────────────────────────┘
                           ▼
                ┌─────────────────────┐
                │  allocate_patrols()  │  zone scores → patrol unit
                │  (fusion.py)         │  allocation (largest-remainder)
                └──────────┬───────────┘
                           ▼
                      app.py (Streamlit dashboard)
```

## Module reference

| File | Purpose |
|---|---|
| `modules/detection.py` | YOLOv8n plug-in point (`YOLOv8ParkingDetector`) + `MockDetector` fallback |
| `modules/ocr.py` | PaddleOCR plug-in point (`PaddleOCRPlateReader`) + `MockPlateReader` fallback |
| `modules/tracker.py` | Lightweight IoU tracker, computes per-vehicle dwell time |
| `modules/historical.py` | Loads precomputed Congestion Impact Scores from real ASTRAM data; matches a live (lat, lon) to its nearest corridor |
| `modules/fusion.py` | Combines live + historical signals into one Enforcement Priority Score; patrol allocation |
| `app.py` | Streamlit dashboard — Historical Intelligence / Live Detection / Fused Enforcement |
| `data/` | Precomputed corridor & police-station impact scores, hourly pattern, trimmed historical event table |

## Wiring in the real CV pipeline

1. Train/export YOLOv8n weights (`best.pt`).
2. In `app.py`, replace `MockDetector()` with:
   ```python
   from modules.detection import YOLOv8ParkingDetector
   detector = YOLOv8ParkingDetector("weights/best.pt")
   ```
3. Feed real frames (`cv2.VideoCapture`) through `detector.detect(frame)`
   instead of `mock_detector.detect()`. Output format is identical
   (`Detection(bbox, vehicle_type, confidence)`), so nothing downstream
   (tracker, fusion, dashboard) needs to change.
4. Optional: wire in `PaddleOCRPlateReader` the same way for plate reads.

## Data honesty note (for your submission writeup)

The provided ASTRAM dataset has no labeled "illegal parking" field. The
Congestion Impact Score uses vehicle-obstruction-type causes
(`vehicle_breakdown`, `congestion`, `others`, `debris`) as the proxy ground
truth — physically, an illegally parked vehicle and a broken-down vehicle
have the same effect on traffic flow: an obstruction occupying the
carriageway. State this assumption explicitly in your documentation.

## Tuning

- Dwell threshold and fusion weights: `modules/fusion.py` (`WEIGHTS`,
  `DWELL_SATURATION_SECONDS`).
- Vehicle-type impact weights: `modules/detection.py` (`VEHICLE_IMPACT_WEIGHT`).
- Corridor-matching radius: `modules/historical.py`
  (`CorridorMatcher.nearest_corridor`, `max_km` param).
