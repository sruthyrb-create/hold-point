"""Day-1 check: does Muse Spark 1.3 read a photo of your stove as structured JSON?

    export MODEL_API_KEY=...            # only after credits show under Billing -> Free credits
    python scripts/first_call.py stove.jpg

Hard cap for this script: $0.50. Expected cost: about 1 cent.
Optional: --sam "stove knob" also sends the photo to SAM 3.1 and saves its raw
output to .holdpoint/sam_raw.txt so the parser can be written against real output
(blocked until sam-3.1 has a price in prices.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint.budget import Ledger  # noqa: E402
from holdpoint.meta_client import BASE_URL, MetaClient, encode_frame  # noqa: E402
from holdpoint.witness_vlm import read_frames  # noqa: E402

ITEMS = {
    "back_left": "back-left burner knob",
    "back_right": "back-right burner knob",
    "front_left": "front-left burner knob",
    "front_right": "front-right burner knob",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--sam", help='noun phrase for SAM 3.1, e.g. "stove knob"')
    a = ap.parse_args()

    frame = cv2.imread(a.image)
    if frame is None:
        raise SystemExit(f"Can't read {a.image}")

    ledger = Ledger(cap_usd=0.50, ledger_file=Path(".holdpoint/first_call_ledger.jsonl"))
    client = MetaClient(ledger=ledger)
    readings = read_frames(client, [frame], [0], ITEMS, tag="first_call")
    for fr in readings:
        for r in fr.items:
            print(f"  {r.item_id:12} visible={r.visible!s:5} state={r.state!s:5} "
                  f"conf={r.confidence:.2f} reason={r.reason_code.value}")
    print("Spark:", ledger.summary())

    if a.sam:
        ledger.check("sam-3.1", expected_cost=ledger.unit_cost("sam-3.1", images=1))
        content = [{"type": "input_text", "text": a.sam},
                   {"type": "input_image", "image_url": encode_frame(frame)}]
        resp = client.client.responses.create(
            model="sam-3.1", input=[{"type": "message", "role": "user", "content": content}])
        ledger.record_units("sam-3.1", images=1, tag="first_call_sam")
        out = Path(".holdpoint/sam_raw.txt")
        out.write_text(resp.model_dump_json(indent=2))
        print(f"SAM raw output saved to {out} (send it to Claude to write the parser).")
        print("Total:", ledger.summary())


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit(f"Set MODEL_API_KEY first (base URL {BASE_URL}).")
    main()
