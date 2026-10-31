"""Witness A: Muse Spark reads each named item from a handful of frames.

One request carries several frames (cheaper than one request per frame) and
asks for a reading per item PER FRAME, so the decision layer still sees a
spread of evidence rather than one summary guess.
"""
from __future__ import annotations

import numpy as np

from .meta_client import MetaClient, encode_frame
from .schemas import FrameReadings, ItemReading, ReasonCode

MODEL = "muse-spark-1.3"

SYSTEM = (
    "You verify the state of household appliance controls for a blind user. "
    "Never guess. If an item is not clearly visible in a frame, set visible=false, state=null, "
    "and give the reason_code (OUT_OF_FRAME, OCCLUDED, TOO_DARK, BLUR, TOO_FAR, GLARE). "
    "Only set visible=true when you can actually see the item's marking or display in that frame."
)

PER_FRAME_SCHEMA = {
    "name": "HoldPointFrames",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["frames"],
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["frame", "items"],
                    "properties": {
                        "frame": {"type": "integer"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["item_id", "visible", "state", "confidence", "reason_code"],
                                "properties": {
                                    "item_id": {"type": "string"},
                                    "visible": {"type": "boolean"},
                                    "state": {"type": ["string", "null"]},
                                    "confidence": {"type": "number"},
                                    "reason_code": {"type": "string",
                                                    "enum": [r.value for r in ReasonCode]},
                                },
                            },
                        },
                    },
                },
            }
        },
    },
}


def build_prompt(items: dict[str, str], n_frames: int, allowed_states: str) -> str:
    lines = "\n".join(f"- {iid}: {desc}" for iid, desc in items.items())
    return (
        f"There are {n_frames} frames, numbered 0 to {n_frames - 1} in the order given. "
        f"For EVERY frame, report EVERY item below.\n"
        f"Items:\n{lines}\n"
        f"Allowed states: {allowed_states}. "
        "confidence is 0 to 1. Use reason_code OK only when visible=true."
    )


def read_frames(client: MetaClient, frames: list[np.ndarray], frame_indices: list[int],
                items: dict[str, str], allowed_states: str = '"off", "on"',
                reasoning_effort: str = "low", tag: str = "") -> list[FrameReadings]:
    """items: {"back_left": "back-left burner knob", ...}"""
    images = [encode_frame(f) for f in frames]
    raw = client.chat_json(MODEL, SYSTEM, build_prompt(items, len(frames), allowed_states),
                           images, PER_FRAME_SCHEMA, reasoning_effort=reasoning_effort, tag=tag)
    out = []
    for fr in raw.get("frames", []):
        i = fr.get("frame")
        if not isinstance(i, int) or not 0 <= i < len(frame_indices):
            continue
        readings = []
        for it in fr.get("items", []):
            if it.get("item_id") not in items:
                continue  # ignore invented items
            try:
                readings.append(ItemReading(**{**it, "confidence": min(max(it["confidence"], 0.0), 1.0)}))
            except Exception:
                continue  # malformed reading = no evidence, never a guess
        out.append(FrameReadings(frame_index=frame_indices[i], witness="vlm", items=readings))
    return out
