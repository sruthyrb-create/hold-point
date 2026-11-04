"""One-time setup: enrol personal items and name the landmarks.

Item: show the item to the camera in 2-4 frames. SAM 3.1 finds it by its
name in each frame; the clearest crops are saved as its enrolment photos; Muse
Spark writes a one-line description if the user didn't give one.

Landmark: point the camera at it once; SAM 3.1 must be able to find it with
the chosen word, otherwise we ask for a different word (a landmark SAM can't
find can never confirm a place).
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from . import sam as sam_mod
from .home import HOME_DIR, Home, Item, Landmark
from .meta_client import encode_frame, prepare_frame
from .witness_spark import crop_context

ITEMS_DIR = HOME_DIR / "items"
DESCRIBE_SCHEMA = {"name": "ItemDescription", "schema": {
    "type": "object", "additionalProperties": False, "required": ["description"],
    "properties": {"description": {"type": "string"}}}}


class EnrolError(ValueError):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not s:
        raise EnrolError("Give the item a name.")
    return s[:30]


def _best(objs):
    if not objs:
        return None
    return max(objs, key=lambda o: ((o.confidence or 0.5), (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])))


def enrol_item(client, home: Home, frames: list[np.ndarray], name: str, sam_phrase: str = "",
               description: str = "", aliases: list[str] | None = None, segment=None, items_dir: Path = ITEMS_DIR) -> Item:
    segment = segment or (lambda c, img, p: sam_mod.segment_image(c, img, p, tag="enrol"))
    iid = slug(name)
    phrase = (sam_phrase or name).strip()
    if not 2 <= len(frames) <= 4:
        raise EnrolError("Show the item in 2 to 4 photos, from different sides.")
    items_dir.mkdir(parents=True, exist_ok=True)
    refs, crops = [], []
    for k, f in enumerate(frames):
        f = prepare_frame(f)
        obj = _best(segment(client, f, phrase))
        if obj is None:
            continue
        crop = crop_context(f, obj.box, factor=1.3, max_side=384)
        path = items_dir / f"{iid}_{k}.jpg"
        cv2.imwrite(str(path), crop)
        refs.append(str(path))
        crops.append(crop)
    if len(refs) < 2:
        raise EnrolError(f"I could only find the {name} in {len(refs)} photo(s) using the word "
                         f"'{phrase}'. Try a plainer word (e.g. 'eyeglasses', 'keys', 'pen') or a plain background.")
    if not description:
        raw = client.chat_json("muse-spark-1.3",
                               "Describe a personal item so it can be told apart from similar items.",
                               f"These photos show the user's {name}. One sentence, under 20 words: type, "
                               "colour, shape, any distinctive marks.",
                               [encode_frame(c) for c in crops], DESCRIBE_SCHEMA, tag="enrol")
        description = str(raw.get("description", "")).strip()[:200] or name
    item = Item(id=iid, name=name.strip().lower(), sam_phrase=phrase, description=description,
                aliases=[a.strip().lower() for a in (aliases or []) if a.strip()], references=refs)
    home.add_item(item)
    home.save()
    return item


def add_landmark(client, home: Home, frame: np.ndarray | None, name: str, room: str,
                 sam_phrase: str = "", segment=None) -> Landmark:
    segment = segment or (lambda c, img, p: sam_mod.segment_image(c, img, p, tag="landmark_setup"))
    phrase = (sam_phrase or name).strip()
    if frame is not None:
        objs = segment(client, prepare_frame(frame), phrase)
        if not objs:
            raise EnrolError(f"I couldn't find a '{phrase}' in that photo. Try another word for it.")
    lm = Landmark(id=slug(f"{room}_{name}"), name=name.strip().lower(), room=room.strip().lower(), sam_phrase=phrase)
    home.add_landmark(lm)
    home.save()
    return lm
