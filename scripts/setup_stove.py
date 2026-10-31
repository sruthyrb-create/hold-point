"""One-time setup: save each knob's OFF position from a photo with ALL knobs off.

    python scripts/setup_stove.py stove.jpg --knobs 4

Cost: one SAM 3.1 image (~$0.0025).
Writes home/profile.json (edit the "name" fields to e.g. "back-left burner"),
home/stove_refs.npz, and home/stove_setup.jpg showing which number is which knob.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint.meta_client import MetaClient, prepare_frame  # noqa: E402
from holdpoint.profile import HOME_DIR, knob_entries, save_stove  # noqa: E402
from holdpoint.witness_sam import knob_crop, segment_image  # noqa: E402


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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    knobs = knob_entries(a.knobs)
    refs = {}
    annotated = frame.copy()
    for entry, obj in zip(knobs, found):          # both left to right
        crop = knob_crop(gray, obj.box)
        if crop is None:
            raise SystemExit(f"{entry['name']} is too close to the photo edge. Step back and retake.")
        refs[entry["id"]] = crop
        entry["setup_box"] = list(obj.box)
        x0, y0, x1, y1 = obj.box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
        cv2.putText(annotated, entry["id"].split("_")[1], (x0, max(30, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

    save_stove(knobs, refs, a.prompt)
    cv2.imwrite(str(HOME_DIR / "stove_setup.jpg"), annotated)
    print(f"Saved {a.knobs} knobs (numbered left to right). See {HOME_DIR / 'stove_setup.jpg'}.")
    print(f"Optional: edit the \"name\" of each knob in {HOME_DIR / 'profile.json'}.")
    print(client.ledger.summary())


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    main()
