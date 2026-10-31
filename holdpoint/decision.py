"""Deterministic decision layer. No AI in this file.

Rule: answer an item only when BOTH witnesses, over several frames,
independently and stably agree. Otherwise abstain with one reason code.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schemas import FrameReadings, ItemReading, ReasonCode, Verdict


@dataclass(frozen=True)
class DecisionConfig:
    min_visible_frames: int = 3     # per witness
    min_agreement: float = 0.8      # share of visible frames voting for the top state
    min_vlm_confidence: float = 0.6


# When we have to pick one reason, prefer the one the user can fix most directly.
REASON_PRIORITY = [
    ReasonCode.TOO_DARK, ReasonCode.GLARE, ReasonCode.BLUR, ReasonCode.OUT_OF_FRAME,
    ReasonCode.OCCLUDED, ReasonCode.TOO_FAR, ReasonCode.LOW_EVIDENCE, ReasonCode.WITNESS_DISAGREE,
]


def _pick_reason(reasons: list[ReasonCode]) -> ReasonCode:
    counts = Counter(r for r in reasons if r is not ReasonCode.OK)
    if not counts:
        return ReasonCode.LOW_EVIDENCE
    top = max(counts.values())
    common = [r for r, c in counts.items() if c >= top * 0.5]
    return min(common, key=REASON_PRIORITY.index)


def _witness_vote(readings: list[ItemReading], cfg: DecisionConfig, is_vlm: bool):
    """Return (state or None, n_visible_frames, reasons)."""
    usable, reasons = [], []
    for r in readings:
        ok = r.visible and r.state is not None and r.reason_code is ReasonCode.OK
        if is_vlm and r.confidence < cfg.min_vlm_confidence:
            ok = False
        if ok:
            usable.append(r.state.strip().lower())
        else:
            reasons.append(r.reason_code if r.reason_code is not ReasonCode.OK else ReasonCode.LOW_EVIDENCE)
    if len(usable) < cfg.min_visible_frames:
        return None, len(usable), reasons
    state, n = Counter(usable).most_common(1)[0]
    if n / len(usable) < cfg.min_agreement:
        return None, len(usable), reasons + [ReasonCode.LOW_EVIDENCE]
    return state, len(usable), reasons


def decide(item_ids: list[str], vlm_frames: list[FrameReadings], geo_frames: list[FrameReadings],
           quality: dict[int, list[ReasonCode]] | None = None,
           cfg: DecisionConfig = DecisionConfig(),
           geometry_items: set[str] | None = None) -> list[Verdict]:
    """geometry_items: items that Witness B can measure (knobs). Displays use a
    second VLM read of the SAM-cropped region instead, passed in geo_frames too."""
    quality = quality or {}
    verdicts = []
    for item in item_ids:
        def collect(frames):
            out = []
            for f in frames:
                bad = quality.get(f.frame_index, [])
                for r in f.items:
                    if r.item_id != item:
                        continue
                    if bad:  # a frame that fails quality checks can't count as evidence
                        r = r.model_copy(update={"visible": False, "reason_code": bad[0]})
                    out.append(r)
            return out

        a_state, a_n, a_reasons = _witness_vote(collect(vlm_frames), cfg, is_vlm=True)
        b_state, b_n, b_reasons = _witness_vote(collect(geo_frames), cfg, is_vlm=False)
        evidence = min(a_n, b_n)

        if a_state is None or b_state is None:
            verdicts.append(Verdict(item_id=item, answered=False, evidence_frames=evidence,
                                    reason_code=_pick_reason(a_reasons + b_reasons),
                                    detail=f"vlm={a_state} geo={b_state}"))
        elif a_state != b_state:
            verdicts.append(Verdict(item_id=item, answered=False, evidence_frames=evidence,
                                    reason_code=ReasonCode.WITNESS_DISAGREE,
                                    detail=f"vlm={a_state} geo={b_state}"))
        else:
            verdicts.append(Verdict(item_id=item, answered=True, state=a_state,
                                    evidence_frames=evidence))
    return verdicts
