"""One-time setup: save each knob's OFF position from a photo with ALL knobs off.

    python scripts/setup_stove.py stove.jpg --knobs 4

Cost: one SAM 3.1 image (~$0.0025).
Writes home/profile.json (edit the "name" fields to e.g. "back-left burner")
and home/stove_setup.jpg showing which number is which knob.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint.meta_client import MetaClient, prepare_frame  # noqa: E402
from holdpoint.profile import HOME_DIR, knob_entries, save_stove  # noqa: E402
from holdpoint.witness_sam import knob_circle, measure_knobs, segment_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="photo with ALL knobs OFF, taken from where you normally stand")
    ap.add_argument("--knobs", type=int, required=True, help="how many knobs the stove has")
    ap.add_argument("--prompt", default="stove knob")
    a = ap.parse_args()

    raw = cv2.imread(a.image)
    if raw is None:
        raise SystemExit(f"Can't read {a.image}")
    frame = prepare_frame(raw)

    client = MetaClient()
    found = segment_image(client, frame, a.prompt, tag="setup")
    if len(found) != a.knobs:
        raise SystemExit(f"SAM found {len(found)} knobs, expected {a.knobs}. "
                         "Retake the photo with every knob clearly in view, or try --prompt \"control knob\".")

    knobs = knob_entries(a.knobs)
    annotated = frame.copy()
    for entry, obj, (direction, peak) in zip(knobs, found, measure_knobs(frame, found)):  # left to right
        if direction is None:
            raise SystemExit(f"Couldn't see {entry['name']}'s grip bar clearly (strength {peak:.1f}). "
                             "Retake in better light, a little closer, holding still.")
        entry["off_direction"] = round(direction, 2)
        entry["setup_box"] = list(obj.box)
        cx, cy, r = knob_circle(obj)
        t = math.radians(direction)
        cv2.line(annotated, (int(cx - math.sin(t) * r), int(cy + math.cos(t) * r)),
                 (int(cx + math.sin(t) * r), int(cy - math.cos(t) * r)), (0, 0, 255), 2)
        x0, y0, x1, y1 = obj.box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
        cv2.putText(annotated, entry["id"].split("_")[1], (x0, max(30, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

    save_stove(knobs, a.prompt)
    cv2.imwrite(str(HOME_DIR / "stove_setup.jpg"), annotated)
    print(f"Saved {a.knobs} knobs (numbered left to right). See {HOME_DIR / 'stove_setup.jpg'}:")
    print("  the red line on each knob should run along its grip bar.")
    print(f"Optional: edit the \"name\" of each knob in {HOME_DIR / 'profile.json'}.")
    print(client.ledger.summary())


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    main()
