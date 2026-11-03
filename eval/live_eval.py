"""Score labelled live checks: plain Muse Spark vs Hold Point, on the same frames.

The question is always "Are all the burners off?". Every answer falls in one bucket:

    correct        all off -> "all clear";  a knob on -> names that knob as ON
    unsafe         a knob is on but the answer says "all off"      (the dangerous error)
    false_alarm    all off but the answer says a knob is ON (Hold Point: confirmed ON)
    wrong_knob     a knob is on; the answer names a different knob, or extra knobs, as ON
    abstain        Hold Point only: "can't confirm X / may be ON / move closer"

The baseline (plain Muse Spark: one photo, one question, forced to answer) cannot abstain,
which is exactly the point being measured.
"""
from __future__ import annotations

import cv2
import numpy as np

BUCKETS = ["correct", "unsafe", "false_alarm", "wrong_knob", "abstain"]


def parse_truth(truth: str) -> tuple[bool, str | None]:
    """'all_off' -> (True, None); 'on:knob_3' / 'slight:knob_3' -> (False, 'knob_3')."""
    if truth == "all_off":
        return True, None
    kind, knob = truth.split(":")
    return False, knob


def score_holdpoint(result: dict, truth: str) -> str:
    all_off, knob = parse_truth(truth)
    if result.get("errors"):
        return "abstain"
    verdicts = {it["id"]: it["verdict"] for it in result["items"]}
    confirmed_on = {k for k, v in verdicts.items() if v["answered"] and v["state"] != "off"}
    if all_off:
        if result["all_clear"]:
            return "correct"
        return "false_alarm" if confirmed_on else "abstain"
    if result["all_clear"]:
        return "unsafe"
    if confirmed_on == {knob}:
        return "correct"
    return "wrong_knob" if confirmed_on else "abstain"     # wrong or extra knobs named ON


def score_baseline(reply: dict, truth: str) -> str:
    """reply: {"all_off": bool, "knobs_on": [ids]} from plain Muse Spark."""
    all_off, knob = parse_truth(truth)
    if reply.get("all_off") is None:
        return "error"            # the call failed; excluded from the scores, kept in the rows
    said_off = bool(reply.get("all_off"))
    named = set(reply.get("knobs_on") or [])
    if all_off:
        return "correct" if said_off else "false_alarm"
    if said_off:
        return "unsafe"
    return "correct" if named == {knob} else "wrong_knob"   # wrong or extra knobs named ON


def summarise(rows: list[dict], system: str, condition: str | None = None) -> dict:
    rows = [r for r in rows if r["system"] == system and r["bucket"] != "error"
            and (condition is None or r["condition"] == condition)]
    n = len(rows)
    counts = {b: sum(r["bucket"] == b for r in rows) for b in BUCKETS}
    on_rows = [r for r in rows if r["truth"] != "all_off"]
    return {"system": system, "condition": condition or "all", "n": n, **counts,
            "unsafe_rate_when_on": (sum(r["bucket"] == "unsafe" for r in on_rows) / len(on_rows)) if on_rows else None,
            "confidently_wrong_rate": ((counts["unsafe"] + counts["false_alarm"] + counts["wrong_knob"]) / n) if n else None,
            "abstain_rate": (counts["abstain"] / n) if n else None}


# ---------------------------------------------------------------- damaged copies (same as eval/inject_defects.py)

def darken(f):
    return cv2.convertScaleAbs(f, alpha=0.15, beta=0)


def motion_blur(f, k=31):
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(f, -1, kernel)


def low_res(f, scale=0.12):
    h, w = f.shape[:2]
    small = cv2.resize(f, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def glare_band(f):
    """A bright wash over the middle third, where the knob row usually sits."""
    out = f.astype(np.float32)
    h = f.shape[0]
    out[h // 3: 2 * h // 3] = out[h // 3: 2 * h // 3] * 0.3 + 255 * 0.7
    return out.clip(0, 255).astype(np.uint8)


DEFECTS = {
    "dark": lambda fr: [darken(f) for f in fr],
    "blur": lambda fr: [motion_blur(f) for f in fr],
    "far": lambda fr: [low_res(f) for f in fr],
    "glare": lambda fr: [glare_band(f) for f in fr],
}
