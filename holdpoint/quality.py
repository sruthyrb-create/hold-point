"""Deterministic per-frame image-quality checks (no AI).

These give the decision layer reasons that don't depend on either model,
so a dark or shaky frame is caught even if both models guess confidently.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import ReasonCode

DARK_MEAN = 45.0        # mean grey level below this = too dark
BLUR_LAPLACIAN = 60.0   # Laplacian variance below this = blurry
GLARE_FRACTION = 0.04   # share of near-white pixels in the region
MIN_ITEM_PIXELS = 24    # item box shorter side below this = too far


def frame_reasons(frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> list[ReasonCode]:
    """Return quality problems for a frame, or for an item's box (x0, y0, x1, y1)."""
    img = frame_bgr
    if roi is not None:
        x0, y0, x1, y1 = roi
        img = frame_bgr[max(y0, 0):y1, max(x0, 0):x1]
        if img.size == 0:
            return [ReasonCode.OUT_OF_FRAME]
        if min(img.shape[:2]) < MIN_ITEM_PIXELS:
            return [ReasonCode.TOO_FAR]

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reasons: list[ReasonCode] = []
    if grey.mean() < DARK_MEAN:
        reasons.append(ReasonCode.TOO_DARK)
    if cv2.Laplacian(grey, cv2.CV_64F).var() < BLUR_LAPLACIAN:
        reasons.append(ReasonCode.BLUR)
    if (grey > 245).mean() > GLARE_FRACTION:
        reasons.append(ReasonCode.GLARE)
    return reasons
