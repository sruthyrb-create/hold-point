"""One live stove check: a few frames -> both witnesses -> decision -> answer.

SAM 3.1 runs first (one call per frame, in parallel) and finds the knobs. Muse Spark
then reads close-ups of the KNOB ROW cut from each frame, instead of the whole
kitchen, where knobs are often too small to see (31 Oct: Spark said TOO_FAR on every
knob in every frame). SAM only says WHERE the row is; Spark still judges each
knob's state on its own, so the two readings stay independent. The close-up is a
full-width band, so a knob can't be cut out of it by a wrong box.

Returns everything the app shows: the short answer, each knob's verdict, what each
witness read in each frame, the Meta model calls made (time taken), the cost of
the check, and an overlay image of what SAM 3.1 found and what the geometry measured.

Cost per check with 4 frames: 4 SAM 3.1 images ($0.01) + one Muse Spark call
with 4 images (about $0.01). Roughly 2 cents.
"""
from __future__ import annotations

import base64
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from .budget import BudgetExceeded
from .decision import DecisionConfig, decide
from .geometry import state_from_angle
from .guidance import all_off_answer
from .meta_client import MetaClient, prepare_frame
from .quality import frame_reasons
from .schemas import FrameReadings, Verdict
from .witness_sam import MODEL as SAM_MODEL
from .witness_sam import (SamObject, assign_knobs, bar_direction, knob_circle, pick_row, read_knobs,
                          segment_image)
from .witness_vlm import MODEL as SPARK_MODEL
from .witness_vlm import read_frames

UNAVAILABLE = "Can't check right now. Try again in a moment."

# OFF window for the MIDDLE of several agreeing frames (single frames keep +/-6).
# Measured 31 Oct after phone setup: untouched knobs' middle turn ranged -0.9..+8.1 deg
# (bias from small viewpoint changes, not noise); the smallest real turn measured was
# +13 deg (slight turn, must stay "unsure") and real ON settings 28-43 deg. ON still
# starts beyond 14 deg, and Muse Spark must independently say OFF too.
CONSENSUS_OFF_DEG = 8.0

GREEN, RED, YELLOW = (60, 200, 60), (40, 40, 230), (0, 210, 240)


def _overlay(frame: np.ndarray, found: list[SamObject], geo: FrameReadings, ids: list[str]) -> str:
    """Draw what Witness B saw on one frame: SAM's knob outline + the measured bar."""
    img = frame.copy()
    by_id = {r.item_id: r for r in geo.items}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for iid, obj in assign_knobs(found, ids, frame.shape[1]).items():
        reading = by_id.get(iid)
        state = reading.state if reading else None
        colour = GREEN if state == "off" else RED if state == "on" else YELLOW
        cx, cy, r = knob_circle(obj)
        cv2.circle(img, (int(cx), int(cy)), int(r), colour, 3)
        direction, _ = bar_direction(gray, cx, cy, r)
        if direction is not None:
            t = math.radians(direction)
            cv2.line(img, (int(cx - math.sin(t) * r), int(cy + math.cos(t) * r)),
                     (int(cx + math.sin(t) * r), int(cy - math.cos(t) * r)), colour, 3)
        label = iid.split("_")[-1]
        if reading and reading.angle_deg is not None:
            label += f" {reading.angle_deg:+.0f}"
        cv2.putText(img, label, (int(cx - r), int(cy - r - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    h, w = img.shape[:2]
    if w > 720:
        img = cv2.resize(img, (720, int(h * 720 / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def knob_row_band(frame: np.ndarray, found: list[SamObject], n_knobs: int,
                  out_width: int = 1280) -> np.ndarray | None:
    """Full-width horizontal strip around the knob row, enlarged. None if SAM found nothing."""
    row = pick_row(found, n_knobs)
    if not row:
        return None
    d = float(np.median([min(o.box[2] - o.box[0], o.box[3] - o.box[1]) for o in row]))
    h = frame.shape[0]
    y0 = max(0, int(min(o.box[1] for o in row) - 1.2 * d))   # room for the printed dial above
    y1 = min(h, int(max(o.box[3] for o in row) + 1.2 * d))   # and the labels below
    band = frame[y0:y1]
    if band.shape[0] < 8:
        return None
    scale = out_width / band.shape[1]
    if scale > 1:
        band = cv2.resize(band, (out_width, int(band.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    return band


def _unwrap(turns: list[float]) -> list[float]:
    """Bar turns are modulo 180 (-90..90]. Put them on one side of the wrap so
    that e.g. +88 and -89 count as 3 degrees apart, not 177."""
    ref = turns[0]
    return [ref + ((t - ref + 90.0) % 180.0 - 90.0) for t in turns]


def geometry_consensus(geo: list[FrameReadings], ids: list[str], max_spread: float = 8.0,
                       min_frames: int = 3) -> list[FrameReadings]:
    """Judge each knob on the MIDDLE of its per-frame turns instead of each frame alone.

    Only applied when the frames are consistent (all measured turns within
    `max_spread` degrees of each other, in at least `min_frames` frames). Then one
    shaky frame near the edge of the OFF window can't force a re-check. If the frames
    are NOT consistent (e.g. one frame saw a big turn), the original per-frame
    readings are kept, so the decision layer still abstains and still warns "may be ON".
    """
    out = [f.model_copy(deep=True) for f in geo]
    for iid in ids:
        measured = [(fi, r) for fi, f in enumerate(out) for r in f.items
                    if r.item_id == iid and r.angle_deg is not None and r.visible]
        if len(measured) < min_frames:
            continue
        turns = _unwrap([r.angle_deg for _, r in measured])
        if max(turns) - min(turns) > max_spread:
            continue
        middle = float(np.median(turns))
        middle = (middle + 90.0) % 180.0 - 90.0
        consensus = state_from_angle(iid, middle, 0.0, period=180.0, off_tolerance=CONSENSUS_OFF_DEG)
        for fi, r in measured:
            f = out[fi]
            f.items = [consensus.model_copy(update={"angle_deg": r.angle_deg}) if x is r else x for x in f.items]
    return out


def run_stove_check(client: MetaClient, frames: list[np.ndarray], stove: dict, refs: dict[str, float],
                    cfg: DecisionConfig = DecisionConfig(),
                    segment: Callable = segment_image, tag: str = "live") -> dict:
    """frames: BGR images from the phone. stove/refs: from profile.load_stove()."""
    start = time.perf_counter()
    spent_before = client.ledger.spent()
    frames = [prepare_frame(f) for f in frames]
    n = len(frames)
    knobs = stove["knobs"]
    ids = [k["id"] for k in knobs]
    names = {k["id"]: k["name"] for k in knobs}
    positions = {k["id"]: k.get("position") for k in knobs}
    quality = {i: frame_reasons(f) for i, f in enumerate(frames)}

    calls: list[dict] = []
    errors: list[str] = []

    def sam_one(i: int):
        t = time.perf_counter()
        found = [replace(o, frame=i) for o in segment(client, frames[i], stove["sam_prompt"], tag=tag)]
        calls.append({"model": SAM_MODEL, "job": f"find knobs in frame {i + 1}",
                      "seconds": round(time.perf_counter() - t, 2), "result": f"{len(found)} found"})
        return found

    found_by_frame: list[list[SamObject]] = [[] for _ in range(n)]
    vlm: list[FrameReadings] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        sam_futs = [ex.submit(sam_one, i) for i in range(n)]
        for i, fut in enumerate(sam_futs):
            try:
                found_by_frame[i] = fut.result()
            except BudgetExceeded:
                raise
            except Exception as e:  # a failed call is missing evidence, never a guess
                errors.append(f"{SAM_MODEL} frame {i + 1}: {type(e).__name__}: {e}"[:200])

    # Witness A reads close-ups of the knob row (whole frame if SAM found nothing there).
    bands = [knob_row_band(frames[i], found_by_frame[i], len(ids)) for i in range(n)]
    spark_images = [b if b is not None else frames[i] for i, b in enumerate(bands)]
    n_close = sum(b is not None for b in bands)
    descriptions = {k["id"]: k["description"] for k in knobs}
    if n_close:
        descriptions = {iid: d + " (images are close-ups of the stove's knob row)" for iid, d in descriptions.items()}
    t = time.perf_counter()
    try:
        vlm = read_frames(client, spark_images, list(range(n)), descriptions, tag=tag)
        calls.append({"model": SPARK_MODEL,
                      "job": f"read every knob in {n} frames, one request ({n_close} knob-row close-ups cut where SAM found the knobs)",
                      "seconds": round(time.perf_counter() - t, 2),
                      "result": f"{sum(len(f.items) for f in vlm)} readings"})
    except BudgetExceeded:
        raise
    except Exception as e:
        errors.append(f"{SPARK_MODEL}: {type(e).__name__}: {e}"[:200])

    geo = [read_knobs(frames[i], i, found_by_frame[i], refs) for i in range(n)]
    geo_judged = geometry_consensus(geo, ids, min_frames=cfg.min_visible_frames)
    verdicts: list[Verdict] = decide(ids, vlm, geo_judged, quality={i: q for i, q in quality.items() if q}, cfg=cfg)
    answer = UNAVAILABLE if errors else all_off_answer(verdicts, names, positions)

    # Per-knob evidence for the model-calls panel.
    items = []
    for iid in ids:
        spark_reads = [{"frame": f.frame_index + 1, "state": r.state, "confidence": r.confidence,
                        "reason": r.reason_code.value}
                       for f in vlm for r in f.items if r.item_id == iid]
        sam_reads = [{"frame": g.frame_index + 1, "state": r.state, "turn_deg": r.angle_deg,
                      "reason": r.reason_code.value}
                     for g in geo for r in g.items if r.item_id == iid]
        judged = [r for g in geo_judged for r in g.items if r.item_id == iid and r.angle_deg is not None]
        middle = float(np.median(_unwrap([r.angle_deg for r in judged]))) if judged else None
        v = next(v for v in verdicts if v.item_id == iid)
        items.append({"id": iid, "name": names[iid], "spark": spark_reads, "sam": sam_reads,
                      "sam_middle_deg": None if middle is None else round(middle, 1),
                      "verdict": {"answered": v.answered, "state": v.state, "reason": v.reason_code.value,
                                  "possibly_on": v.possibly_on, "evidence_frames": v.evidence_frames}})

    best = max(range(n), key=lambda i: len(found_by_frame[i])) if n else 0
    return {
        "answer": answer,
        "all_clear": not errors and all(v.answered and v.state == "off" for v in verdicts),
        "danger": any((v.answered and v.state != "off") or v.possibly_on for v in verdicts),
        "items": items,
        "calls": sorted(calls, key=lambda c: c["model"]),
        "errors": errors,
        "quality": {str(i + 1): [q.value for q in qs] for i, qs in quality.items() if qs},
        "frames": n,
        "seconds": round(time.perf_counter() - start, 2),
        "cost_usd": round(client.ledger.spent() - spent_before, 4),
        "overlay": _overlay(frames[best], found_by_frame[best], geo[best], ids) if n else "",
    }
