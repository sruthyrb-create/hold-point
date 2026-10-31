"""Witness B tests: real SAM output parsing + rotation matching on synthetic knobs."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from holdpoint.schemas import ReasonCode
from holdpoint.witness_sam import (CROP, SamObject, assign_knobs, knob_crop, measure_turn,
                                   parse_sam_text, read_knobs)

FIXTURE = Path(__file__).parent / "fixtures" / "sam_four_knobs.txt"


def test_parses_real_sam_output():
    objs = parse_sam_text(FIXTURE.read_text())
    assert len(objs) == 4
    assert [o.box[0] for o in objs] == sorted(o.box[0] for o in objs)   # left to right
    for o in objs:
        x0, y0, x1, y1 = o.box
        assert o.mask.shape == (y1 - y0, x1 - x0)
        assert 0.7 < o.mask.mean() < 0.9   # round knob fills ~pi/4 of its box


def knob_image(angle_deg, symmetric=False, size=CROP, blur=True):
    """Grey knob with a grip bar and (unless symmetric) a white pointer dot.
    Drawn 4x larger then shrunk, so it is accurately centred and anti-aliased."""
    S, k = size * 4, 4
    img = np.full((S, S), 60, np.uint8)
    c = ((S - 1) / 2, (S - 1) / 2)
    P = lambda x, y: (int(round(x * 16)), int(round(y * 16)))   # 4 fractional bits
    cv2.circle(img, P(*c), int(S * 0.45 * 16), 150, -1, cv2.LINE_AA, 4)
    t = math.radians(angle_deg)
    dx, dy = math.sin(t), -math.cos(t)
    r = S * 0.38
    cv2.line(img, P(c[0] - dx * r, c[1] - dy * r), P(c[0] + dx * r, c[1] + dy * r), 90, 9 * k, cv2.LINE_AA, 4)
    if not symmetric:
        cv2.circle(img, P(c[0] + dx * r * 0.85, c[1] + dy * r * 0.85), 6 * k * 16, 250, -1, cv2.LINE_AA, 4)
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return cv2.GaussianBlur(out, (0, 0), 2.0) if blur else out


@pytest.mark.parametrize("true_angle", [0, 4, 45, 90, 180, 270, 330])
def test_measure_turn(true_angle):
    angle, score, _ = measure_turn(knob_image(0), knob_image(true_angle))
    assert angle is not None and score > 0.8
    assert abs(((angle - true_angle + 180) % 360) - 180) <= 4


def test_symmetric_knob_is_ambiguous_so_abstains():
    # A bar with no pointer looks identical at 0 and 180 degrees; 180 would be ON.
    angle, _, _ = measure_turn(knob_image(0, symmetric=True), knob_image(0, symmetric=True))
    assert angle is None


def test_tolerates_box_jitter():
    """SAM boxes wobble: crop the current knob 4 px off-centre and 8% too big."""
    from holdpoint.witness_sam import knob_crop
    frame = np.full((400, 400), 40, np.uint8)
    big = cv2.resize(knob_image(90, blur=False).astype(np.uint8), (160, 160))
    frame[100:260, 100:260] = big
    ref_frame = frame.copy()
    ref_frame[100:260, 100:260] = cv2.resize(knob_image(0, blur=False).astype(np.uint8), (160, 160))
    ref = knob_crop(ref_frame, (100, 100, 260, 260))
    cur = knob_crop(frame, (104, 97, 271, 267))   # shifted and enlarged box
    angle, score, _ = measure_turn(ref, cur)
    assert angle is not None and abs(angle - 90) <= 6


def test_unrelated_image_abstains():
    noise = np.random.default_rng(1).integers(0, 255, (CROP, CROP)).astype(np.float32)
    angle, score, _ = measure_turn(knob_image(0), noise)
    assert angle is None


def obj(x0, x1, y0=100, y1=180):
    return SamObject(0, str(x0), (x0, y0, x1, y1), None)


def test_assign_knobs_edges():
    ids = ["k1", "k2", "k3", "k4"]
    four = [obj(100, 180), obj(300, 380), obj(800, 880), obj(1000, 1080)]
    assert list(assign_knobs(four, ids, 1280)) == ids
    # cut off on the left: leftmost visible knob touches the left edge
    two_left = [obj(2, 80), obj(200, 280)]
    assert list(assign_knobs(two_left, ids, 1280)) == ["k3", "k4"]
    # cut off on the right
    two_right = [obj(900, 980), obj(1200, 1279)]
    assert list(assign_knobs(two_right, ids, 1280)) == ["k1", "k2"]
    # fewer knobs, no edge clue: refuse to guess identities
    assert assign_knobs([obj(500, 580)], ids, 1280) == {}


def paste_knob(frame, box, angle):
    x0, y0, x1, y1 = box
    k = cv2.resize(knob_image(angle, blur=False).astype(np.uint8), (x1 - x0, y1 - y0))
    frame[y0:y1, x0:x1] = cv2.cvtColor(k, cv2.COLOR_GRAY2BGR)


def test_read_knobs_end_to_end():
    boxes = [(100, 400, 196, 496), (300, 400, 396, 496)]
    ref_frame = np.full((720, 1280, 3), 40, np.uint8)
    for b in boxes:
        paste_knob(ref_frame, b, 0)
    gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    refs = {"left": knob_crop(gray, boxes[0]), "right": knob_crop(gray, boxes[1])}

    now = np.full((720, 1280, 3), 40, np.uint8)
    paste_knob(now, boxes[0], 0)      # still off
    paste_knob(now, boxes[1], 120)    # turned on
    found = [SamObject(0, "a", boxes[0], None), SamObject(0, "b", boxes[1], None)]
    r = {x.item_id: x for x in read_knobs(now, 0, found, refs).items}
    assert r["left"].state == "off"
    assert r["right"].state == "on"

    r = {x.item_id: x for x in read_knobs(now, 0, found[:1], refs).items}  # one knob unseen
    assert r["left"].reason_code is ReasonCode.OUT_OF_FRAME and r["right"].reason_code is ReasonCode.OUT_OF_FRAME
