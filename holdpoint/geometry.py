"""Witness B: turn SAM 3.1 masks into knob angles and an on/off state.

SAM only says *where* things are. This module does the measuring:
the knob's angle is the direction from the knob's centre to its pointer
mark, compared with the OFF angle captured during home setup.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .schemas import ItemReading, ReasonCode

OFF_TOLERANCE_DEG = 12.0   # within this of OFF = off
AMBIGUOUS_BAND_DEG = 8.0   # between tol and tol+band = don't trust it


def centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def pointer_angle(knob_mask: np.ndarray, pointer_mask: np.ndarray) -> Optional[float]:
    """Angle in degrees of the pointer around the knob centre.

    0 = pointing up (12 o'clock), increasing clockwise, range [0, 360).
    """
    k, p = centroid(knob_mask), centroid(pointer_mask)
    if k is None or p is None:
        return None
    dx, dy = p[0] - k[0], p[1] - k[1]
    if math.hypot(dx, dy) < 1.0:
        return None
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def read_knob(item_id: str, knob_mask: Optional[np.ndarray], pointer_mask: Optional[np.ndarray],
              off_angle_deg: float) -> ItemReading:
    if knob_mask is None or not knob_mask.any():
        return ItemReading(item_id=item_id, visible=False, reason_code=ReasonCode.OUT_OF_FRAME)
    if pointer_mask is None or not pointer_mask.any():
        # Knob found but its pointer isn't: something is covering it or it's too small.
        return ItemReading(item_id=item_id, visible=False, reason_code=ReasonCode.OCCLUDED)

    angle = pointer_angle(knob_mask, pointer_mask)
    if angle is None:
        return ItemReading(item_id=item_id, visible=False, reason_code=ReasonCode.TOO_FAR)
    return state_from_angle(item_id, angle, off_angle_deg)


def state_from_angle(item_id: str, angle: float, off_angle_deg: float = 0.0) -> ItemReading:
    """Shared rule: a narrow OFF window, an 'unsure' band around it, everything else ON."""
    off_by = abs(angle_diff(angle, off_angle_deg))
    if off_by <= OFF_TOLERANCE_DEG:
        state, conf = "off", 1.0 - off_by / (OFF_TOLERANCE_DEG + AMBIGUOUS_BAND_DEG)
    elif off_by <= OFF_TOLERANCE_DEG + AMBIGUOUS_BAND_DEG:
        return ItemReading(item_id=item_id, visible=True, state=None, confidence=0.0,
                           reason_code=ReasonCode.LOW_EVIDENCE, angle_deg=angle)
    else:
        state, conf = "on", min(1.0, off_by / 45.0)
    return ItemReading(item_id=item_id, visible=True, state=state, confidence=round(conf, 3),
                       angle_deg=round(angle, 1))
