"""Witness B: SAM 3.1 finds each knob; geometry measures how far it has turned.

How the angle is measured
-------------------------
Knob pointers look different on every stove, so instead of asking SAM to find
the pointer, we compare each knob with the SAME knob in the user's all-off
reference photo (captured once at setup). We rotate the reference crop in 2°
steps and find the rotation that best matches the current crop (normalised
cross-correlation inside the knob's circle). That rotation is the knob's
angle away from OFF.

Safety rules built in:
- A weak best match (occluded, blurred, wrong object) -> abstain.
- Two equally good matches far apart (e.g. a symmetric bar knob that looks
  the same at 0° and 180°) -> abstain. 180° would be ON, so we never guess.
- If SAM finds fewer knobs than the profile expects, we only name knobs
  whose identity is certain from the frame edge; the rest are OUT_OF_FRAME.

SAM output is decoded with Meta's official parser, meta-sam-parser.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import angle_diff, state_from_angle
from .schemas import FrameReadings, ItemReading, ReasonCode

MODEL = "sam-3.1"
CROP = 96                 # knob crops are resized to CROP x CROP
INNER_RADIUS = 0.46       # compare pixels inside this fraction of the crop
MIN_MATCH = 0.45          # best correlation below this = can't trust the match
RIVAL_MARGIN = 0.06       # a rival peak this close to the best = ambiguous
RIVAL_MIN_SEPARATION = 30.0
EDGE_FRACTION = 0.06      # a knob this close to the image edge may be cut off
BLUR_SIGMA = 2.0          # smoothing applied to every crop before matching


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


# ---------------------------------------------------------------- angle by rotation matching

def knob_crop(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Square crop centred on the knob, resized to CROP x CROP (float32)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    a, b = int(round(cx - half)), int(round(cy - half))
    c, d = int(round(cx + half)), int(round(cy + half))
    if a < 0 or b < 0 or c > gray.shape[1] or d > gray.shape[0] or c - a < 8:
        return None
    crop = cv2.resize(gray[b:d, a:c], (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Smooth so a few pixels of SAM box jitter don't change the match much.
    return cv2.GaussianBlur(crop, (0, 0), BLUR_SIGMA)


_yy, _xx = np.mgrid[:CROP, :CROP]
_INNER = ((_xx - (CROP - 1) / 2) ** 2 + (_yy - (CROP - 1) / 2) ** 2) <= (INNER_RADIUS * CROP) ** 2


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[_INNER], b[_INNER]
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


def rotation_scores(reference: np.ndarray, current: np.ndarray, step: float = 2.0) -> dict[float, float]:
    """Score for 'current = reference turned clockwise by angle'."""
    centre = ((CROP - 1) / 2, (CROP - 1) / 2)
    scores = {}
    for ang in np.arange(0, 360, step):
        m = cv2.getRotationMatrix2D(centre, -float(ang), 1.0)  # negative = clockwise on screen
        rotated = cv2.warpAffine(reference, m, (CROP, CROP), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        scores[float(ang)] = _ncc(rotated, current)
    return scores


def measure_turn(reference: np.ndarray, current: np.ndarray) -> tuple[float | None, float, float]:
    """Returns (angle from OFF or None if untrustworthy, best score, rival score)."""
    scores = rotation_scores(reference, current)
    best = max(scores, key=scores.get)
    rival = max((s for a, s in scores.items() if abs(angle_diff(a, best)) > RIVAL_MIN_SEPARATION),
                default=-1.0)
    if scores[best] < MIN_MATCH or rival > scores[best] - RIVAL_MARGIN:
        return None, scores[best], rival
    return best, scores[best], rival


# ---------------------------------------------------------------- identity + reading

def assign_knobs(found: list[SamObject], expected_ids: list[str], image_width: int) -> dict[str, SamObject]:
    """Map detected knobs (left to right) to profile ids (left to right).

    Same count -> one to one. Fewer found -> only safe if the missing knobs must be
    past an image edge (a knob is touching that edge, the other side is clear).
    """
    found = sorted(found, key=lambda o: o.centre[0])
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


def read_knobs(frame_bgr: np.ndarray, frame_index: int, found: list[SamObject],
               references: dict[str, np.ndarray]) -> FrameReadings:
    """references: item_id -> OFF reference crop (from the home profile), in left-to-right order."""
    ids = list(references)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    assigned = assign_knobs([o for o in found if o.frame == frame_index], ids, frame_bgr.shape[1])

    readings = []
    for iid in ids:
        obj = assigned.get(iid)
        if obj is None:
            readings.append(ItemReading(item_id=iid, visible=False, reason_code=ReasonCode.OUT_OF_FRAME))
            continue
        crop = knob_crop(gray, obj.box)
        if crop is None:
            readings.append(ItemReading(item_id=iid, visible=False, reason_code=ReasonCode.OUT_OF_FRAME))
            continue
        if min(obj.box[2] - obj.box[0], obj.box[3] - obj.box[1]) < 24:
            readings.append(ItemReading(item_id=iid, visible=False, reason_code=ReasonCode.TOO_FAR))
            continue
        angle, score, _ = measure_turn(references[iid], crop)
        if angle is None:
            reason = ReasonCode.OCCLUDED if score < MIN_MATCH else ReasonCode.LOW_EVIDENCE
            readings.append(ItemReading(item_id=iid, visible=score >= MIN_MATCH, reason_code=reason))
            continue
        readings.append(state_from_angle(iid, angle, 0.0))
    return FrameReadings(frame_index=frame_index, witness="geometry", items=readings)
