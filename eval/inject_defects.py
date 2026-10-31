"""Create labelled broken copies of a clean walkthrough video.

Usage:
    python eval/inject_defects.py walkthrough.mp4 walkthrough.truth.json out_dir/

Ground-truth JSON (written by hand once per clean video):
{
  "items": {
    "back_left":  {"state": "on",  "window": [2.0, 4.5], "box": [410, 120, 520, 230]},
    "back_right": {"state": "off", "window": [3.0, 6.0], "box": [600, 130, 700, 230]}
  }
}
  window = seconds when the item is in view; box = its pixel box in that window
  (a rough box is fine; it's only used to occlude it).

Each output clip gets a <name>.label.json saying, per item, either
"expect": <true state>  (a correct answer is required; abstaining is allowed but counted)
or "expect": "ABSTAIN" with the reason the item can't be verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_video(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read any frames from {path}")
    return frames, fps


def write_video(path: Path, frames, fps):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


# ---- whole-clip degradations: every item becomes unverifiable ----

def darken(f):
    return cv2.convertScaleAbs(f, alpha=0.15, beta=0)


def motion_blur(f, k=31):
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(f, -1, kernel)


def low_res(f, scale=0.08):
    h, w = f.shape[:2]
    small = cv2.resize(f, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def glare(f, box=None):
    out = f.copy()
    h, w = f.shape[:2]
    if box:
        x0, y0, x1, y1 = box
        cx, cy, r = (x0 + x1) // 2, (y0 + y1) // 2, int(max(x1 - x0, y1 - y0) * 0.9)
    else:
        cx, cy, r = w // 2, h // 2, min(h, w) // 3
    cv2.circle(out, (cx, cy), r, (255, 255, 255), -1)
    return cv2.GaussianBlur(out, (0, 0), 3)


def shake(frames, px=25, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for f in frames:
        dx, dy = rng.integers(-px, px + 1, size=2)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(motion_blur(cv2.warpAffine(f, m, (f.shape[1], f.shape[0])), k=21))
    return out


WHOLE_CLIP = {
    "dark": ("TOO_DARK", lambda fr: [darken(f) for f in fr]),
    "blur": ("BLUR", lambda fr: shake(fr)),
    "far": ("TOO_FAR", lambda fr: [low_res(f) for f in fr]),
    "glare": ("GLARE", lambda fr: [glare(f) for f in fr]),
}


def frame_range(window, fps, n):
    return max(0, int(window[0] * fps)), min(n, int(window[1] * fps) + 1)


def make_variants(video: str, truth_path: str, out_dir: str) -> list[dict]:
    frames, fps = read_video(video)
    truth = json.loads(Path(truth_path).read_text())["items"]
    stem = Path(video).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def emit(name, new_frames, labels, defect):
        clip = out / f"{stem}__{name}.mp4"
        write_video(clip, new_frames, fps)
        label = {"source": Path(video).name, "defect": defect, "items": labels}
        (out / f"{stem}__{name}.label.json").write_text(json.dumps(label, indent=2))
        made.append({"clip": clip.name, **label})

    truth_labels = {k: {"expect": v["state"]} for k, v in truth.items()}

    # 1. control
    emit("control", frames, truth_labels, "none")

    # 2. whole-clip degradations
    for name, (reason, fn) in WHOLE_CLIP.items():
        emit(name, fn(frames), {k: {"expect": "ABSTAIN", "reason": reason} for k in truth}, name)

    # 3. per-item: occlude it, or cut it out entirely
    for item, info in truth.items():
        a, b = frame_range(info["window"], fps, len(frames))

        if "box" in info:
            occl = [f.copy() for f in frames]
            x0, y0, x1, y1 = info["box"]
            for i in range(a, b):
                cv2.rectangle(occl[i], (x0, y0), (x1, y1), (20, 20, 20), -1)
            labels = dict(truth_labels)
            labels[item] = {"expect": "ABSTAIN", "reason": "OCCLUDED"}
            emit(f"occlude_{item}", occl, labels, "occlude")

        # Cut: drop the frames where the item is visible. Other items that are only
        # visible inside that window also disappear, so relabel them too.
        cut = frames[:a] + frames[b:]
        if len(cut) >= int(fps):  # keep at least ~1s of video
            labels = {}
            for k, v in truth.items():
                ka, kb = frame_range(v["window"], fps, len(frames))
                still_seen = (ka < a) or (kb > b)
                labels[k] = truth_labels[k] if still_seen else {"expect": "ABSTAIN", "reason": "OUT_OF_FRAME"}
            emit(f"cut_{item}", cut, labels, "cut")

    (out / f"{stem}__manifest.json").write_text(json.dumps(made, indent=2))
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("truth")
    p.add_argument("out_dir")
    a = p.parse_args()
    made = make_variants(a.video, a.truth, a.out_dir)
    print(f"Wrote {len(made)} labelled clips to {a.out_dir}")
