"""Where is the item relative to a landmark, in the same image? Plain geometry, no AI.

There is no SLAM or depth here: positions are relations in the image ("on",
"left of", "right of", "near"), which is all a spoken answer needs. Image
left/right is the camera's view, so it matches what the user faced when the
item was seen; answers say "left of the stove" only from that viewpoint.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]   # x0, y0, x1, y1

RELATIONS = ("on", "left_of", "right_of", "near")
ON_OVERLAP = 0.5      # this share of the item's box inside the landmark's box -> "on"
NEAR_FACTOR = 1.5     # gap up to 1.5x the landmark's width still anchors the item


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap_share(item: Box, lm: Box) -> float:
    ix = max(0.0, min(item[2], lm[2]) - max(item[0], lm[0]))
    iy = max(0.0, min(item[3], lm[3]) - max(item[1], lm[1]))
    a = _area(item)
    return ix * iy / a if a else 0.0


def mask_overlap_share(item: Box, lm: Box, lm_mask) -> float:
    """Share of the item's box that lies on the landmark's actual outline (SAM mask,
    which covers exactly the landmark's box). A box alone over-claims "on" for big
    landmarks like a stove, whose rectangle includes empty space."""
    x0, y0 = max(item[0], lm[0]), max(item[1], lm[1])
    x1, y1 = min(item[2], lm[2]), min(item[3], lm[3])
    a = _area(item)
    if x1 <= x0 or y1 <= y0 or not a:
        return 0.0
    h, w = lm_mask.shape
    sx, sy = w / max(1e-6, lm[2] - lm[0]), h / max(1e-6, lm[3] - lm[1])
    r0, r1 = int((y0 - lm[1]) * sy), max(int((y0 - lm[1]) * sy) + 1, int(np.ceil((y1 - lm[1]) * sy)))
    c0, c1 = int((x0 - lm[0]) * sx), max(int((x0 - lm[0]) * sx) + 1, int(np.ceil((x1 - lm[0]) * sx)))
    covered = float(lm_mask[r0:r1, c0:c1].sum()) / (sx * sy)
    return min(1.0, covered / a)


def relation(item: Box, lm: Box, lm_mask=None) -> str | None:
    """"on" / "left_of" / "right_of" / "near", or None if the landmark is too far
    away in the image to describe where the item is. With the landmark's SAM mask,
    "on" needs the item to lie on the landmark itself, not just inside its box."""
    share = mask_overlap_share(item, lm, lm_mask) if lm_mask is not None else overlap_share(item, lm)
    if share >= ON_OVERLAP:
        return "on"
    if lm_mask is not None and overlap_share(item, lm) >= ON_OVERLAP:
        # inside the box but off the outline: beside it, on whichever side is nearer
        icx = (item[0] + item[2]) / 2
        cols = np.flatnonzero(lm_mask.any(axis=0))
        if cols.size:
            sx = lm_mask.shape[1] / max(1e-6, lm[2] - lm[0])
            left_edge, right_edge = lm[0] + cols[0] / sx, lm[0] + (cols[-1] + 1) / sx
            if icx < left_edge:
                return "left_of"
            if icx > right_edge:
                return "right_of"
        return "near"
    lm_w = lm[2] - lm[0]
    lm_h = lm[3] - lm[1]
    icx = (item[0] + item[2]) / 2
    icy = (item[1] + item[3]) / 2
    if item[2] <= lm[0]:
        gap_x = lm[0] - item[2]
    elif item[0] >= lm[2]:
        gap_x = item[0] - lm[2]
    else:
        gap_x = 0.0
    gap_y = max(0.0, lm[1] - item[3], item[1] - lm[3])
    if gap_x > NEAR_FACTOR * lm_w or gap_y > NEAR_FACTOR * lm_h:
        return None
    if gap_x > 0 and icx < lm[0]:
        return "left_of"
    if gap_x > 0 and icx > lm[2]:
        return "right_of"
    return "near"


def relations_agree(spark: str | None, geometry: str | None) -> bool:
    """Witness agreement on the place. "near" is the weaker claim, so it agrees with
    any specific relation geometry supports; opposite sides never agree."""
    if spark is None or geometry is None:
        return False
    if spark == geometry:
        return True
    return "near" in (spark, geometry)


def place_phrase(relation_: str, landmark_name: str, room: str) -> str:
    where = {"on": f"on the {landmark_name}", "left_of": f"left of the {landmark_name}",
             "right_of": f"right of the {landmark_name}", "near": f"near the {landmark_name}"}[relation_]
    return f"{where}, in the {room}" if room else where


def direction_in_frame(item: Box, width: int, height: int) -> str:
    """For the final "found it" step: where the item is in the current view."""
    cx = (item[0] + item[2]) / 2 / width
    share = _area(item) / (width * height)
    side = "ahead, slightly left" if cx < 0.4 else "ahead, slightly right" if cx > 0.6 else "straight ahead"
    if cx < 0.2:
        side = "to your left"
    elif cx > 0.8:
        side = "to your right"
    dist = "close" if share > 0.05 else "further away" if share < 0.005 else ""
    return f"{side}{', ' + dist if dist else ''}"
