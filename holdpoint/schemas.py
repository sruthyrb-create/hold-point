"""Data types shared by both witnesses and the decision layer."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReasonCode(str, Enum):
    OK = "OK"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    OCCLUDED = "OCCLUDED"
    TOO_DARK = "TOO_DARK"
    BLUR = "BLUR"
    TOO_FAR = "TOO_FAR"
    GLARE = "GLARE"
    WITNESS_DISAGREE = "WITNESS_DISAGREE"
    LOW_EVIDENCE = "LOW_EVIDENCE"


class ItemReading(BaseModel):
    """One witness's reading of one item in one frame.

    Witness A (Muse Spark) fills this via structured output.
    Witness B (SAM 3.1 geometry) fills it from masks + angle maths.
    """

    item_id: str
    visible: bool
    state: Optional[str] = None  # "off", "on", or a value such as "375"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason_code: ReasonCode = ReasonCode.OK
    angle_deg: Optional[float] = None  # geometry witness only


class FrameReadings(BaseModel):
    frame_index: int
    witness: str  # "vlm" or "geometry"
    items: list[ItemReading]


class Verdict(BaseModel):
    item_id: str
    answered: bool
    state: Optional[str] = None
    reason_code: ReasonCode = ReasonCode.OK
    evidence_frames: int = 0
    possibly_on: bool = False   # unconfirmed, but at least one witness saw it ON
    detail: str = ""


# JSON schema handed to Muse Spark as response_format (Chat Completions).
VLM_RESPONSE_SCHEMA = {
    "name": "HoldPointReadings",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
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
                        "reason_code": {"type": "string", "enum": [r.value for r in ReasonCode]},
                    },
                },
            }
        },
    },
}
