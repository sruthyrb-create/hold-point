"""Check one photo of the stove with both witnesses and print Hold Point's answer.

    python scripts/check_photo.py photo.jpg

Cost: ~1 cent (one SAM image + one Muse Spark call). Repeat runs on the same
photo are free (cached for Spark).

Single-photo mode needs only 1 agreeing frame per witness. The real app uses
several frames from video, which is stricter.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint.decision import DecisionConfig, decide  # noqa: E402
from holdpoint.guidance import all_off_answer  # noqa: E402
from holdpoint.meta_client import MetaClient, prepare_frame  # noqa: E402
from holdpoint.profile import load_stove  # noqa: E402
from holdpoint.quality import frame_reasons  # noqa: E402
from holdpoint.witness_sam import read_knobs, segment_image  # noqa: E402
from holdpoint.witness_vlm import read_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    a = ap.parse_args()

    raw = cv2.imread(a.image)
    if raw is None:
        raise SystemExit(f"Can't read {a.image}")
    frame = prepare_frame(raw)
    stove, refs = load_stove()
    knobs = stove["knobs"]
    ids = [k["id"] for k in knobs]

    client = MetaClient()
    quality = frame_reasons(frame)
    found = segment_image(client, frame, stove["sam_prompt"], tag="check")
    geo = read_knobs(frame, 0, found, refs)
    vlm = read_frames(client, [frame], [0], {k["id"]: k["description"] for k in knobs}, tag="check")

    print(f"Frame quality: {[q.value for q in quality] or 'OK'}   SAM found {len(found)} knob(s)")
    print(f"{'knob':10} {'Spark (A)':22} {'SAM geometry (B)':28}")
    a_by = {r.item_id: r for f in vlm for r in f.items}
    b_by = {r.item_id: r for r in geo.items}
    for k in ids:
        ra, rb = a_by.get(k), b_by.get(k)
        sa = f"{ra.state} ({ra.confidence:.2f})" if ra and ra.state else (ra.reason_code.value if ra else "missing")
        sb = (f"{rb.state} ({rb.angle_deg}° from OFF)" if rb.state else rb.reason_code.value)
        print(f"{k:10} {sa:22} {sb:28}")

    verdicts = decide(ids, vlm, [geo], quality={0: quality} if quality else None,
                      cfg=DecisionConfig(min_visible_frames=1))
    names = {k["id"]: k["name"] for k in knobs}
    positions = {k["id"]: k["position"] for k in knobs}
    print("\nHold Point says:", all_off_answer(verdicts, names, positions))
    print(client.ledger.summary())


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    main()
