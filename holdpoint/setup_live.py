"""Stove setup from the phone: capture every knob's OFF direction from the same
phone, distance and angle the user will check from (removes perspective offset).

Uses several frames and takes each knob's middle direction; refuses if the frames
disagree, if SAM doesn't find exactly the expected number of knobs, or if a bar
can't be seen. Keeps the names already in the profile.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .check import _unwrap
from .meta_client import MetaClient, prepare_frame
from .profile import HOME_DIR, knob_entries
from .witness_sam import knob_circle, measure_knobs, pick_row, segment_image

MAX_SPREAD_DEG = 10.0  # a handheld frame wobbles about +/-4 deg (31 Oct); the MIDDLE of 6 frames is saved


class SetupFailed(ValueError):
    pass


def setup_from_frames(client: MetaClient, frames: list[np.ndarray], n_knobs: int,
                      prompt: str = "stove knob", segment: Callable = segment_image,
                      home: Path = HOME_DIR) -> dict:
    frames = [prepare_frame(f) for f in frames]
    attempt = home / "setup_attempts" / datetime.now().strftime("%Y%m%d-%H%M%S")
    attempt.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    per_knob: list[list[float]] = [[] for _ in range(n_knobs)]
    boxes: list[tuple] = [None] * n_knobs
    good_frames = 0
    best = None
    for i, frame in enumerate(frames):
        cv2.imwrite(str(attempt / f"frame{i + 1}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        raw = [replace(o, frame=i) for o in segment(client, frame, prompt, tag="setup")]
        found = pick_row(raw, n_knobs)
        if len(found) != n_knobs:
            notes.append(f"frame {i + 1}: found {len(raw)} knob-like objects, expected {n_knobs}")
            continue
        found = sorted(found, key=lambda o: o.centre[0])
        dirs = measure_knobs(frame, found)
        unclear = [k + 1 for k, (d, _) in enumerate(dirs) if d is None]
        if unclear:
            notes.append(f"frame {i + 1}: grip bar unclear on knob {', '.join(map(str, unclear))} "
                         f"(strength {', '.join(f'{p:.1f}' for _, p in dirs)})")
            continue
        good_frames += 1
        best = best or (frame, found, [d for d, _ in dirs])
        for k, ((d, _), obj) in enumerate(zip(dirs, found)):
            per_knob[k].append(d)
            boxes[k] = obj.box

    def fail(message: str):
        (attempt / "notes.txt").write_text(message + "\n" + "\n".join(notes) + "\n")
        raise SetupFailed(message + (" Details: " + "; ".join(notes) if notes else ""))

    if good_frames < 2:
        fail(f"Couldn't see all {n_knobs} knobs clearly. Turn on a light, "
             "hold the phone where you normally stand, keep every knob in view, and try again.")

    off_dirs = []
    for k, dirs in enumerate(per_knob):
        u = _unwrap(dirs)
        notes.append(f"knob {k + 1} directions: {', '.join(f'{d:.1f}' for d in u)}")
        if len(u) >= 4:   # one wobbly frame out of several shouldn't sink the setup: drop the farthest
            mid = float(np.median(u))
            u.remove(max(u, key=lambda d: abs(d - mid)))
        if max(u) - min(u) > MAX_SPREAD_DEG:
            fail(f"Knob {k + 1} looked different between frames. Hold still and try again.")
        off_dirs.append(round(float(np.median(u)) % 180.0, 2))
    (attempt / "notes.txt").write_text("saved\n" + "\n".join(notes) + "\n")

    # keep names/descriptions the user may have edited; replace only the measurements
    profile_path = home / "profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    old = profile.get("stove", {}).get("knobs", [])
    knobs = old if len(old) == n_knobs else knob_entries(n_knobs)
    for entry, d, box in zip(knobs, off_dirs, boxes):
        entry["off_direction"] = d
        entry["setup_box"] = list(box)
    profile["stove"] = {"sam_prompt": prompt, "knobs": knobs}
    home.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2))

    frame, found, dirs = best
    img = frame.copy()
    for k, (obj, d) in enumerate(zip(found, dirs)):
        cx, cy, r = knob_circle(obj)
        t = math.radians(d)
        cv2.circle(img, (int(cx), int(cy)), int(r), (60, 200, 60), 3)
        cv2.line(img, (int(cx - math.sin(t) * r), int(cy + math.cos(t) * r)),
                 (int(cx + math.sin(t) * r), int(cy - math.cos(t) * r)), (40, 40, 230), 3)
        cv2.putText(img, str(k + 1), (int(cx - r), int(cy - r - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 200, 60), 2)
    cv2.imwrite(str(home / "stove_setup.jpg"), img)
    import base64
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (720, int(img.shape[0] * 720 / img.shape[1]))),
                           [cv2.IMWRITE_JPEG_QUALITY, 70])
    return {"message": f"Setup saved: {n_knobs} knobs, all OFF, from {good_frames} frames. "
                       "Check that each red line runs along its knob's grip bar.",
            "off_directions": off_dirs, "frames_used": good_frames,
            "overlay": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""}
