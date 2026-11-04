"""Hide-and-seek evaluation: score answers as correct / wrong / not sure.

Truth file (eval/truth.json), one trial per hiding place:

[
  {"trial": "glasses_01", "item": "glasses", "landmark": "kitchen_stove", "relation": "left_of",
   "frames": "clips/glasses_01/"}            # folder of walk-through frames, in time order
]

Headline: confidently-wrong rate (said a place, and it was the wrong one).
"Not sure" is fine only when it's honest; we also report how often the system
was not sure when the item WAS visible with a landmark (coverage).
"""
from __future__ import annotations

from dataclasses import dataclass

CONFIDENT = {"confirmed", "stale", "two_places"}


@dataclass
class Outcome:
    trial: str
    said_landmarks: list[str]      # places the answer committed to (empty = not sure / never seen)
    status: str
    truth_landmark: str


def classify(o: Outcome) -> str:
    """two_places counts as correct if the truth is one of the two (it told the user
    both); a single wrong place is "wrong"."""
    if o.status not in CONFIDENT or not o.said_landmarks:
        return "not_sure"
    return "correct" if o.truth_landmark in o.said_landmarks else "wrong"


def summarise(outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    if not n:
        return {"n": 0}
    labels = [classify(o) for o in outcomes]
    c = {k: labels.count(k) for k in ("correct", "wrong", "not_sure")}
    return {"n": n, **c,
            "confident_wrong_rate": round(c["wrong"] / n, 3),
            "correct_rate": round(c["correct"] / n, 3),
            "not_sure_rate": round(c["not_sure"] / n, 3),
            "precision_when_committed": round(c["correct"] / max(1, c["correct"] + c["wrong"]), 3)}
