"""Experiment: which SAM 3.1 wording outlines the knob's grip bar best?

    python scripts/try_sam_prompts.py stove.jpg on.jpg slight.jpg

Cost: photos x prompts x $0.0025 (3 photos x 4 prompts = 3 cents).
Saves an overlay image per photo and prompt in home/prompt_tests/ (git-ignored),
and prints what SAM found.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint.meta_client import MetaClient, prepare_frame  # noqa: E402
from holdpoint.profile import HOME_DIR  # noqa: E402
from holdpoint.witness_sam import segment_image  # noqa: E402

DEFAULT_PROMPTS = ["knob handle", "black bar on knob", "knob grip", "knob pointer"]
COLOURS = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 220, 220), (255, 0, 255), (255, 255, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    a = ap.parse_args()

    out_dir = HOME_DIR / "prompt_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = MetaClient()
    print(f"About to make {len(a.images) * len(a.prompts)} SAM calls "
          f"(~${len(a.images) * len(a.prompts) * 0.0025:.3f}).")

    for path in a.images:
        raw = cv2.imread(path)
        if raw is None:
            print(f"Can't read {path}, skipping.")
            continue
        frame = prepare_frame(raw)
        for prompt in a.prompts:
            found = segment_image(client, frame, prompt, tag="prompt_test")
            overlay = frame.copy()
            for i, o in enumerate(found):
                x0, y0, x1, y1 = o.box
                colour = COLOURS[i % len(COLOURS)]
                if o.mask is not None:
                    region = overlay[y0:y0 + o.mask.shape[0], x0:x0 + o.mask.shape[1]]
                    tint = np.zeros_like(region)
                    tint[:] = colour
                    m = o.mask[:region.shape[0], :region.shape[1]]
                    region[m] = (0.45 * region[m] + 0.55 * tint[m]).astype(np.uint8)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
            name = f"{Path(path).stem}__{prompt.replace(' ', '_')}.jpg"
            cv2.imwrite(str(out_dir / name), overlay)
            boxes = ", ".join(f"({o.box[0]},{o.box[1]})-({o.box[2]},{o.box[3]})" for o in found)
            print(f"{Path(path).name:12} {prompt!r:22} {len(found)} found: {boxes}")

    print(f"\nOverlays saved in {out_dir}")
    print(client.ledger.summary())


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    main()
