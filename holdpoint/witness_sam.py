"""Witness B: SAM 3.1 finds each knob; geometry measures which way its grip bar points.

How the angle is measured
-------------------------
Stove knobs usually have a raised grip bar. On glossy knobs the metal reflects
differently every time the knob or the camera moves, so comparing pixels is
unreliable (we tried; see docs/engineering-notes.md). What stays reliable is
the bar's two long straight edges: the knob's round rim has edges pointing in
every direction, while the bar's edges all point one way. So we:

1. take the knob's region from SAM 3.1 (its mask; circle-fitted, so a pointer
   tip sticking out doesn't shift the centre),
2. build a histogram of edge directions inside the knob, weighted by edge
   strength, and take the peak as the bar's direction,
3. compare it with the same knob's direction in the user's all-off setup photo.

Safety rules built in:
- No clear peak (blur, a hand in the way, a knob without a bar) -> abstain.
- Fewer knobs found than expected -> only name knobs whose identity is certain
  from the image edge; the rest are OUT_OF_FRAME.

Known limit: a straight bar looks the same pointing up (OFF) and pointing
straight down (half a turn), so this witness can't tell those apart. Hold Point
only says "off" when Witness A independently agrees.

SAM output is decoded with Meta's official parser, meta-sam-parser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import state_from_angle
from .schemas import FrameReadings, ItemReading, ReasonCode

MODEL = "sam-3.1"
INNER = 0.8               # use edges inside this fraction of the knob radius (skip the rim)
BINS = 90                 # 2-degree histogram bins over 0-180
MIN_PEAK = 2.8            # peak / mean of edge histogram: real bars 3.8-5.6, plain dome 1.7
MIN_RADIUS_PX = 12        # smaller than this = too far away to measure
EDGE_FRACTION = 0.06      # a knob this close to the image edge may be cut off


@dataclass
class SamObject:
    frame: int
    object_id: str
    box: tuple[int, int, int, int]       # x0, y0, x1, y1 in the sent image's pixels
    mask: np.ndarray | None              # bool array covering `box`
    confidence: float | None = None

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2


# ---------------------------------------------------------------- parsing

def parse_sam_text(text: str, video: bool = False) -> list[SamObject]:
    import meta_sam_parser as sp

    fmt = sp.video_segmentation_format() if video else sp.image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text)
    result = parser.finish(sp.CompletedOutcome()).result

    boxes, masks = {}, {}
    for r in result.records:
        frame = r.frame.frame_index if getattr(r, "frame", None) is not None else 0
        key = (frame, r.object_id)
        if isinstance(r, sp.SegmentationBoxRecord):
            boxes[key] = (r.left, r.top, r.right, r.bottom, r.confidence)
        elif isinstance(r, sp.SegmentationMaskRecord):
            raster = np.frombuffer(sp.decode_mask_to_raster(r.mask), np.uint8)
            masks[key] = raster.reshape(r.mask.height, r.mask.width).astype(bool)

    out = []
    for (frame, oid), (x0, y0, x1, y1, conf) in boxes.items():
        out.append(SamObject(frame, oid, (int(x0), int(y0), int(x1), int(y1)),
                             masks.get((frame, oid)), conf))
    return sorted(out, key=lambda o: (o.frame, o.centre[0]))


def segment_image(client, image_bgr: np.ndarray, prompt: str, tag: str = "") -> list[SamObject]:
    """One SAM 3.1 call on one image (cost: 1 image). `image_bgr` must be the exact
    image whose pixel coordinates you'll use afterwards (see meta_client.prepare_frame)."""
    from .meta_client import encode_frame

    ledger = client.ledger
    ledger.check(MODEL, expected_cost=ledger.unit_cost(MODEL, images=1))
    content = [{"type": "input_text", "text": prompt},
               {"type": "input_image", "image_url": encode_frame(image_bgr)}]
    resp = client.client.responses.create(
        model=MODEL, input=[{"type": "message", "role": "user", "content": content}])
    ledger.record_units(MODEL, images=1, tag=tag)
    return parse_sam_text(resp.output_text)


# ---------------------------------------------------------------- knob geometry

def knob_circle(obj: SamObject) -> tuple[float, float, float]:
    """Centre (x, y) and radius in image pixels. Fits a circle to the mask outline,
    ignoring the farthest 20% of points (a protruding pointer tip)."""
    x0, y0, x1, y1 = obj.box
    if obj.mask is not None and obj.mask.any():
        cnts, _ = cv2.findContours(obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        if len(pts) >= 12:
            def fit(p):
                a = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
                cx, cy, c = np.linalg.lstsq(a, (p ** 2).sum(1), rcond=None)[0]
                return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 1.0))
            cx, cy, r = fit(pts)
            d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            cx, cy, r = fit(pts[d <= np.percentile(d, 80)])
            return x0 + cx, y0 + cy, r
    r = min(x1 - x0, y1 - y0) / 2       # no mask: the shorter side is the least affected by a tip
    return (x0 + x1) / 2, (y0 + y1) / 2, r


def bar_direction(gray: np.ndarray, cx: float, cy: float, r: float) -> tuple[float | None, float]:
    """Direction of the knob's grip bar in degrees (0 = vertical, clockwise, mod 180),
    and the histogram peak strength. Returns (None, peak) if there's no clear bar."""
    h = int(math.ceil(r * 1.3))
    xa, ya = int(round(cx)) - h, int(round(cy)) - h
    if xa < 0 or ya < 0 or xa + 2 * h > gray.shape[1] or ya + 2 * h > gray.shape[0]:
        return None, 0.0
    crop = gray[ya:ya + 2 * h, xa:xa + 2 * h]
    crop = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(crop).astype(np.float32)
    # Remove gradual shading (light falling across the dome), keep sharp edges like the bar's.
    crop = cv2.GaussianBlur(crop, (0, 0), 1.5) - cv2.GaussianBlur(crop, (0, 0), max(2.0, r / 5))
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
    inside = np.hypot(xx - (cx - xa), yy - (cy - ya)) <= INNER * r
    # an edge runs perpendicular to its gradient: direction (-gy, gx); angle from vertical, clockwise
    ang = np.degrees(np.arctan2(-gy, -gx)) % 180.0
    hist, _ = np.histogram(ang[inside], bins=BINS, range=(0, 180), weights=np.hypot(gx, gy)[inside])
    if hist.sum() <= 0:
        return None, 0.0
    kernel = np.array([1, 2, 3, 2, 1], float) / 9.0
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], kernel, "valid")
    i = int(smooth.argmax())
    peak = float(smooth[i] / smooth.mean())
    if peak < MIN_PEAK:
        return None, peak
    # refine the peak to sub-bin precision with a parabola through its neighbours
    l, c, rgt = smooth[(i - 1) % BINS], smooth[i], smooth[(i + 1) % BINS]
    offset = 0.5 * (l - rgt) / (l - 2 * c + rgt) if (l - 2 * c + rgt) != 0 else 0.0
    return ((i + 0.5 + offset) * 180.0 / BINS) % 180.0, peak


# ---------------------------------------------------------------- identity + reading

def pick_row(found: list[SamObject], n: int) -> list[SamObject]:
    """If SAM found MORE than n knob-like things (e.g. the knob on a pan lid), keep the
    n that sit in one horizontal row with similar sizes, like a stove's knob row.
    Returns `found` unchanged if there's no clear row (the caller then refuses to guess)."""
    if len(found) <= n or n < 2:
        return found
    by_y = sorted(found, key=lambda o: o.centre[1])
    best, best_score = None, None
    for i in range(len(by_y) - n + 1):
        group = by_y[i:i + n]
        sizes = [min(o.box[2] - o.box[0], o.box[3] - o.box[1]) for o in group]
        d = float(np.median(sizes))
        y_spread = (group[-1].centre[1] - group[0].centre[1]) / d       # in knob diameters
        size_spread = (max(sizes) - min(sizes)) / d
        score = y_spread + size_spread
        if best_score is None or score < best_score:
            best, best_score = group, score
    ys = [o.centre[1] for o in best]
    d = float(np.median([min(o.box[2] - o.box[0], o.box[3] - o.box[1]) for o in best]))
    if max(ys) - min(ys) > 0.75 * d:          # not a clean row: don't pick
        return found
    rest = [o for o in found if o not in best]
    # the row must be clearly separate from the extras (an extra on the row's line is ambiguous)
    if any(abs(o.centre[1] - float(np.median(ys))) < 0.75 * d for o in rest):
        return found
    return sorted(best, key=lambda o: o.centre[0])


def assign_knobs(found: list[SamObject], expected_ids: list[str], image_width: int) -> dict[str, SamObject]:
    """Map detected knobs (left to right) to profile ids (left to right).

    Same count -> one to one. Fewer found -> only safe if the missing knobs must be
    past an image edge (a knob is touching that edge, the other side is clear).
    """
    found = sorted(pick_row(found, len(expected_ids)), key=lambda o: o.centre[0])
    n, k = len(expected_ids), len(found)
    if k == n:
        return dict(zip(expected_ids, found))
    if k == 0 or k > n:
        return {}
    edge = EDGE_FRACTION * image_width
    near_left = found[0].box[0] <= edge
    near_right = found[-1].box[2] >= image_width - edge
    if near_left and not near_right:      # cut off on the left: we see the rightmost k
        return dict(zip(expected_ids[n - k:], found))
    if near_right and not near_left:      # cut off on the right: we see the leftmost k
        return dict(zip(expected_ids[:k], found))
    return {}                             # can't tell which knobs these are


def measure_knobs(frame_bgr: np.ndarray, found: list[SamObject]) -> list[tuple[float | None, float]]:
    """Bar direction and peak strength for each found knob (used at setup)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return [bar_direction(gray, *knob_circle(o)) for o in found]


def read_knobs(frame_bgr: np.ndarray, frame_index: int, found: list[SamObject],
               references: dict[str, float]) -> FrameReadings:
    """references: item_id -> the knob's bar direction at OFF (from setup), left to right."""
    ids = list(references)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    assigned = assign_knobs([o for o in found if o.frame == frame_index], ids, frame_bgr.shape[1])

    readings = []
    for iid in ids:
        obj = assigned.get(iid)
        if obj is None:
            readings.append(ItemReading(item_id=iid, visible=False, reason_code=ReasonCode.OUT_OF_FRAME))
            continue
        cx, cy, r = knob_circle(obj)
        if r < MIN_RADIUS_PX:
            readings.append(ItemReading(item_id=iid, visible=False, reason_code=ReasonCode.TOO_FAR))
            continue
        direction, peak = bar_direction(gray, cx, cy, r)
        if direction is None:
            reason = ReasonCode.OUT_OF_FRAME if peak == 0.0 else ReasonCode.LOW_EVIDENCE
            readings.append(ItemReading(item_id=iid, visible=False, reason_code=reason))
            continue
        reading = state_from_angle(iid, direction, references[iid], period=180.0)
        # report the turn relative to OFF, not the raw direction
        turn = (direction - references[iid] + 90.0) % 180.0 - 90.0
        readings.append(reading.model_copy(update={"angle_deg": round(turn, 1)}))
    return FrameReadings(frame_index=frame_index, witness="geometry", items=readings)
