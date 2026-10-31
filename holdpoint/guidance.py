"""Turn verdicts into short, action-first spoken answers (target: <= ~10 words)."""
from __future__ import annotations

from .schemas import ReasonCode, Verdict

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

FIXED_ACTION = {
    ReasonCode.OCCLUDED: "Something's covering it.",
    ReasonCode.TOO_DARK: "Turn on a light.",
    ReasonCode.BLUR: "Hold still.",
    ReasonCode.TOO_FAR: "Move closer.",
    ReasonCode.GLARE: "Tilt the phone slightly.",
    ReasonCode.WITNESS_DISAGREE: "Hold still, checking again.",
    ReasonCode.LOW_EVIDENCE: "Hold still, checking again.",
}


def direction_for(position: str | None) -> str:
    """Map a home-profile position like 'back-left' to a camera move."""
    if not position:
        return "Pan slowly across the stove."
    p = position.lower()
    if "left" in p:
        return "Move the phone left."
    if "right" in p:
        return "Move the phone right."
    if "back" in p or "top" in p:
        return "Tilt the phone up."
    if "front" in p or "bottom" in p:
        return "Tilt the phone down."
    return "Pan slowly across the stove."


def action_for(v: Verdict, position: str | None) -> str:
    if v.reason_code is ReasonCode.OUT_OF_FRAME:
        return direction_for(position)
    return FIXED_ACTION.get(v.reason_code, "Hold still.")


def all_off_answer(verdicts: list[Verdict], names: dict[str, str],
                   positions: dict[str, str] | None = None) -> str:
    """Answer 'are all the burners off?'. Danger first, then gaps, then good news."""
    positions = positions or {}
    on = [v for v in verdicts if v.answered and v.state != "off"]
    off = [v for v in verdicts if v.answered and v.state == "off"]
    unknown = [v for v in verdicts if not v.answered]

    if on:
        listed = " and ".join(names.get(v.item_id, v.item_id) for v in on[:2])
        tail = " Others unconfirmed." if unknown else ""
        return f"No. {listed} {'is' if len(on) == 1 else 'are'} ON.{tail}"
    if not unknown:
        n = NUMBER_WORDS.get(len(off), str(len(off)))
        return f"Yes, all {n} off." if len(off) > 1 else "Yes, it's off."
    first = unknown[0]
    lead = f"{NUMBER_WORDS.get(len(off), len(off)).capitalize()} off. " if off else ""
    name = names.get(first.item_id, first.item_id)
    return f"{lead}Can't confirm {name}. {action_for(first, positions.get(first.item_id))}"


def value_answer(v: Verdict, expected: str | None, name: str, position: str | None = None) -> str:
    """Answer 'is the oven set to 350?'."""
    if not v.answered:
        return f"Can't read the {name}. {action_for(v, position)}"
    if expected is None or v.state == expected.lower():
        return f"It shows {v.state}."
    return f"It shows {v.state}, not {expected}."
