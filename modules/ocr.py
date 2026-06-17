"""
ocr.py
------
Plug-in point for PaddleOCR plate reading. Lowest-priority component --
the system is fully functional and demoable without it (dwell-time +
historical fusion already answer the core problem statement). Wire this
in last, only if time permits.
"""

import random
import string


class PaddleOCRPlateReader:
    """
    Real reader. Lazy-imports paddleocr.

    Usage:
        reader = PaddleOCRPlateReader()
        plate_text = reader.read(cropped_plate_image)  # numpy BGR crop
    """

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR  # noqa: deferred import
        self.engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read(self, plate_crop) -> str:
        result = self.engine.ocr(plate_crop, cls=True)
        if not result or not result[0]:
            return ""
        # Concatenate detected text lines, highest-confidence first
        lines = sorted(result[0], key=lambda r: -r[1][1])
        return "".join(line[1][0] for line in lines).upper().replace(" ", "")


class MockPlateReader:
    """Generates a plausible KA-format plate for demo purposes."""

    def read(self, plate_crop=None) -> str:
        district = random.randint(1, 60)
        letters = "".join(random.choices(string.ascii_uppercase, k=2))
        digits = random.randint(1000, 9999)
        return f"KA{district:02d}{letters}{digits}"
