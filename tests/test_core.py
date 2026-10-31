import json
import math

import cv2
import numpy as np

from eval.inject_defects import make_variants, write_video
from eval.metrics import load_labels, score
from holdpoint.decision import decide
from holdpoint.geometry import angle_diff, pointer_angle, read_knob
from holdpoint.guidance import all_off_answer, value_answer
from holdpoint.quality import frame_reasons
from holdpoint.schemas import FrameReadings, ItemReading, ReasonCode, Verdict


# ---------- geometry ----------

def knob_masks(angle_deg, size=200, r=60, pr=45):
    knob = np.zeros((size, size), np.uint8)
    cv2.circle(knob, (100, 100), r, 1, -1)
    ptr = np.zeros_like(knob)
    t = math.radians(angle_deg)
    x, y = 100 + pr * math.sin(t), 100 - pr * math.cos(t)
    cv2.circle(ptr, (int(round(x)), int(round(y))), 6, 1, -1)
    return knob, ptr


def test_pointer_angle_round_trip():
    for a in [0, 45, 90, 180, 270, 330]:
        k, p = knob_masks(a)
        assert abs(angle_diff(pointer_angle(k, p), a)) < 3


def test_read_knob_states():
    assert read_knob("k", *knob_masks(3), off_angle_deg=0).state == "off"
    assert read_knob("k", *knob_masks(90), off_angle_deg=0).state == "on"
    assert read_knob("k", *knob_masks(15), off_angle_deg=0).reason_code is ReasonCode.LOW_EVIDENCE
    assert read_knob("k", None, None, 0).reason_code is ReasonCode.OUT_OF_FRAME
    k, _ = knob_masks(0)
    assert read_knob("k", k, None, 0).reason_code is ReasonCode.OCCLUDED


# ---------- decision layer ----------

def frames(witness, item, states, visible=True, conf=0.9, reason=ReasonCode.OK):
    return [FrameReadings(frame_index=i, witness=witness, items=[
        ItemReading(item_id=item, visible=visible, state=s, confidence=conf, reason_code=reason)])
        for i, s in enumerate(states)]


def test_agree_answers():
    v = decide(["bl"], frames("vlm", "bl", ["off"] * 5), frames("geometry", "bl", ["off"] * 5))[0]
    assert v.answered and v.state == "off"


def test_disagree_abstains():
    v = decide(["bl"], frames("vlm", "bl", ["off"] * 5), frames("geometry", "bl", ["on"] * 5))[0]
    assert not v.answered and v.reason_code is ReasonCode.WITNESS_DISAGREE and v.possibly_on


def test_confident_vlm_alone_is_not_enough():
    # The core claim: a confident VLM can't answer if geometry never saw the item.
    v = decide(["bl"], frames("vlm", "bl", ["off"] * 5, conf=0.99),
               frames("geometry", "bl", [None] * 5, visible=False, reason=ReasonCode.OUT_OF_FRAME))[0]
    assert not v.answered and v.reason_code is ReasonCode.OUT_OF_FRAME


def test_quality_veto():
    q = {i: [ReasonCode.TOO_DARK] for i in range(5)}
    v = decide(["bl"], frames("vlm", "bl", ["off"] * 5), frames("geometry", "bl", ["off"] * 5), quality=q)[0]
    assert not v.answered and v.reason_code is ReasonCode.TOO_DARK


def test_flicker_abstains():
    v = decide(["bl"], frames("vlm", "bl", ["off", "on", "off", "on", "off"]),
               frames("geometry", "bl", ["off"] * 5))[0]
    assert not v.answered


# ---------- spoken answers ----------

NAMES = {"bl": "back-left", "br": "back-right", "fl": "front-left", "fr": "front-right"}
POS = {"bl": "back-left", "br": "back-right", "fl": "front-left", "fr": "front-right"}


def test_answers_are_short_and_action_first():
    all_off = [Verdict(item_id=k, answered=True, state="off") for k in NAMES]
    assert all_off_answer(all_off, NAMES) == "Yes, all four off."

    gap = all_off[1:] + [Verdict(item_id="bl", answered=False, reason_code=ReasonCode.OUT_OF_FRAME)]
    s = all_off_answer(gap, NAMES, POS)
    assert s == "Three off. Can't confirm back-left. Move the phone left."
    assert len(s.split()) <= 10

    danger = all_off[1:] + [Verdict(item_id="bl", answered=True, state="on")]
    assert all_off_answer(danger, NAMES).startswith("No. back-left is ON")

    maybe = all_off[1:] + [Verdict(item_id="bl", answered=False, possibly_on=True,
                                   reason_code=ReasonCode.WITNESS_DISAGREE)]
    assert all_off_answer(maybe, NAMES) == "Not all clear. back-left may be ON. Check it."

    assert value_answer(Verdict(item_id="oven", answered=True, state="375"), "350", "oven") == \
        "It shows 375, not 350."


# ---------- quality + eval pipeline ----------

def synthetic_frame(brightness=180):
    f = np.full((360, 640, 3), brightness, np.uint8)
    for x in range(0, 640, 20):  # texture so the blur check has edges
        cv2.line(f, (x, 0), (x, 359), (0, 0, 0), 2)
    return f


def test_quality_checks():
    assert frame_reasons(synthetic_frame()) == []
    assert ReasonCode.TOO_DARK in frame_reasons(synthetic_frame(20))
    assert ReasonCode.BLUR in frame_reasons(cv2.GaussianBlur(synthetic_frame(), (0, 0), 8))


def test_defect_injection_and_metrics(tmp_path):
    video = tmp_path / "stove.mp4"
    write_video(video, [synthetic_frame() for _ in range(90)], 30)
    truth = {"items": {
        "bl": {"state": "on", "window": [0.0, 1.0], "box": [50, 50, 150, 150]},
        "br": {"state": "off", "window": [1.5, 3.0], "box": [400, 50, 500, 150]},
    }}
    (tmp_path / "t.json").write_text(json.dumps(truth))
    made = make_variants(str(video), str(tmp_path / "t.json"), str(tmp_path / "out"))
    # control + 4 whole-clip + 2 occlusions + 2 cuts
    assert len(made) == 9

    labels = load_labels(tmp_path / "out")
    cut = labels[("stove__cut_bl.mp4", "bl")]
    assert cut == {"expect": "ABSTAIN", "reason": "OUT_OF_FRAME"}
    assert labels[("stove__cut_bl.mp4", "br")] == {"expect": "off"}

    # A baseline that always says "off" with confidence:
    preds = [{"clip": c, "item": i, "answer": "off", "spoken": "Yes, all the burners are off."}
             for (c, i) in labels]
    s = score(labels, preds)
    assert s["confident_wrong_pct"] > 50 and s["abstain_pct"] == 0
