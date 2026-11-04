"""Witness B: Muse Spark checks whether each SAM candidate is really the user's
enrolled item (compared with the enrolment photos), and names the nearest
landmark from the user's own list.

All candidates in a frame go in ONE request. Spark only picks a landmark from
the given list (or "none"); it can't invent a place. Malformed or invented
answers are dropped, never treated as a sighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .home import HOME_DIR, Home
from .meta_client import encode_frame
from .sam import SamObject

MODEL = "muse-spark-1.3"
MIN_MATCH = 0.7          # Spark confidence needed to accept "this is the user's item"
MAX_REFS = 2             # enrolment photos sent per item
CONTEXT = 3.0            # candidate crop = box enlarged 3x, so the surroundings show

SYSTEM = (
    "You help a blind or low-vision person keep track of their own belongings. "
    "Never guess. Say an object is the user's item only if it clearly matches the enrolment photos "
    "and description (same type, colour, shape). If unsure, is_users_item=false. "
    "For the place, choose only from the given landmark list, and only if that landmark is visible "
    "in the full view next to the object; otherwise landmark='none' and relation='none'."
)


@dataclass
class Candidate:
    item: str
    obj: SamObject


@dataclass
class Match:
    candidate: int
    is_users_item: bool
    confidence: float
    landmark: str | None
    relation: str | None
    reason: str
    surface: str | None = None      # what the item is on/next to, in a few words (one witness only)


def crop_context(frame: np.ndarray, box, factor: float = CONTEXT, max_side: int = 512) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 24) * factor / 2
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    crop = frame[b:d, a:c]
    s = max_side / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return crop


def _schema(landmark_ids: list[str]) -> dict:
    return {"name": "LastSeenMatches", "schema": {
        "type": "object", "additionalProperties": False, "required": ["candidates"],
        "properties": {"candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["candidate", "is_users_item", "confidence", "landmark", "relation", "reason", "surface"],
            "properties": {
                "candidate": {"type": "integer"},
                "is_users_item": {"type": "boolean"},
                "confidence": {"type": "number"},
                "landmark": {"type": "string", "enum": landmark_ids + ["none"]},
                "relation": {"type": "string", "enum": ["on", "left_of", "right_of", "near", "none"]},
                "reason": {"type": "string"},
                "surface": {"type": "string"},
            }}}}}}


def build_request(frame: np.ndarray, candidates: list[Candidate], home: Home,
                  load_image=cv2.imread) -> tuple[str, list[str]]:
    """Image order: enrolment photos per item, then one context crop per candidate,
    then the full view. The prompt says which image is which."""
    images, lines = [], []
    items = sorted({c.item for c in candidates})
    for iid in items:
        it = home.item(iid)
        refs = []
        for p in it.references[:MAX_REFS]:
            p = Path(p)
            if not p.exists():                       # data folder moved (e.g. into Hold Point)
                p = HOME_DIR / "items" / p.name
            img = load_image(str(p)) if p.exists() else None
            if img is not None:
                images.append(encode_frame(img))
                refs.append(len(images) - 1)
        ref_txt = f"enrolment photos: images {refs}" if refs else "no enrolment photo"
        lines.append(f'- item "{iid}": {it.description} ({ref_txt})')
    cand_lines = []
    for i, c in enumerate(candidates):
        images.append(encode_frame(crop_context(frame, c.obj.box)))
        cand_lines.append(f"- candidate {i}: possible {c.item}, close-up in image {len(images) - 1}")
    images.append(encode_frame(frame))
    lm_lines = "\n".join(f'- "{l.id}": {l.name} ({l.room})' for l in home.landmarks) or "- (none set up)"
    text = (
        "Enrolled items:\n" + "\n".join(lines) +
        "\nCandidates found by a segmentation model (each is in the centre of its close-up):\n" +
        "\n".join(cand_lines) +
        f"\nImage {len(images) - 1} is the full camera view.\n"
        "Landmarks (fixed things in the user's home):\n" + lm_lines +
        "\nFor EVERY candidate: is it the user's enrolled item? confidence 0-1. "
        "Which listed landmark is next to it in the full view, and is the object on it, left of it, "
        "right of it (as seen in the image), or just near it? reason: under 12 words. "
        "surface: what the object is lying on or next to, 2-6 plain words (e.g. 'wooden dining table', "
        "'bed beside the pillow'); 'unknown' if you can't tell."
    )
    return text, images


def parse_matches(raw: dict, n_candidates: int, landmark_ids: set[str]) -> list[Match]:
    out = {}
    for m in raw.get("candidates", []):
        try:
            i = int(m["candidate"])
            if not 0 <= i < n_candidates or i in out:
                continue
            lm = m["landmark"] if m["landmark"] in landmark_ids else None
            rel = m["relation"] if lm and m["relation"] in ("on", "left_of", "right_of", "near") else None
            surf = " ".join(str(m.get("surface") or "").split()[:6]).strip(" .").lower()
            out[i] = Match(i, bool(m["is_users_item"]), min(max(float(m["confidence"]), 0.0), 1.0),
                           lm, rel, str(m.get("reason", ""))[:120],
                           surf if surf and surf != "unknown" else None)
        except (KeyError, TypeError, ValueError):
            continue          # malformed = no evidence
    return [out[i] for i in sorted(out)]


def confirm(client, frame: np.ndarray, candidates: list[Candidate], home: Home, tag: str = "") -> list[Match]:
    if not candidates:
        return []
    text, images = build_request(frame, candidates, home)
    ids = [l.id for l in home.landmarks]
    raw = client.chat_json(MODEL, SYSTEM, text, images, _schema(ids), reasoning_effort="low", tag=tag)
    return parse_matches(raw, len(candidates), set(ids))
