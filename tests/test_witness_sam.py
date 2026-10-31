"""Witness B tests: real SAM output parsing + bar-direction geometry on synthetic knobs.

The synthetic knob copies what real glossy knobs look like: a shaded dome with a
dark grip stripe that is OFFSET from the centre (we see the side of a raised fin),
plus a bright highlight along the fin's top edge.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from holdpoint.schemas import ReasonCode
from holdpoint.witness_sam import (SamObject, assign_knobs, bar_direction, knob_circle,
                                   measure_knobs, parse_sam_text, read_knobs)

FIXTURE = Path(__file__).parent / "fixtures" / "sam_four_knobs.txt"


def test_parses_real_sam_output():
    objs = parse_sam_text(FIXTURE.read_text())
    assert len(objs) == 4
    assert [o.box[0] for o in objs] == sorted(o.box[0] for o in objs)   # left to right
    for o in objs:
        x0, y0, x1, y1 = o.box
        assert o.mask.shape == (y1 - y0, x1 - x0)
        assert 0.7 < o.mask.mean() < 0.9   # round knob fills ~pi/4 of its box


def test_circle_fit_on_real_masks():
    for o in parse_sam_text(FIXTURE.read_text()):
        cx, cy, r = knob_circle(o)
        x0, y0, x1, y1 = o.box
        assert 36 < r < 44
        assert abs(cx - (x0 + x1) / 2) < 6 and abs(cy - (y0 + y1) / 2) < 6


def draw_knob(img, cx, cy, r, angle_deg, bar=True, seed=0):
    """Draw a knob onto a BGR image, at 4x then shrunk for anti-aliasing."""
    S = 4
    size = int(r * 3)
    big = np.full((size * S, size * S), 70, np.uint8)
    c = size * S / 2
    yy, xx = np.mgrid[:size * S, :size * S]
    dome = np.hypot(xx - c, yy - c) <= r * S
    shade = (150 + 50 * (xx - c) / (r * S) - 30 * (yy - c) / (r * S)).clip(60, 230)
    big[dome] = shade[dome].astype(np.uint8)
    if bar:
        t = math.radians(angle_deg)
        dx, dy = math.sin(t), -math.cos(t)          # along the bar
        px, py = -dy, dx                            # across the bar
        off = 0.18 * r * S                          # fin side face is offset from centre
        for w, val, shift in [(0.30, 25, off), (0.06, 235, off - 0.17 * r * S)]:  # dark face, bright top edge
            a = (int(c + px * shift - dx * r * S * 0.95), int(c + py * shift - dy * r * S * 0.95))
            b = (int(c + px * shift + dx * r * S * 0.95), int(c + py * shift + dy * r * S * 0.95))
            cv2.line(big, a, b, val, max(1, int(w * r * S)), cv2.LINE_AA)
        big[~dome] = 70
    noise = np.random.default_rng(seed).normal(0, 4, big.shape)
    big = (big + noise).clip(0, 255).astype(np.uint8)
    small = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    img[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    return (int(cx - r), int(cy - r), int(cx + r), int(cy + r))


def frame_with(angles, r=40):
    img = np.full((480, 1280, 3), 70, np.uint8)
    boxes = [draw_knob(img, 200 + 280 * i, 240, r, a, seed=i) for i, a in enumerate(angles)]
    return img, [SamObject(0, str(i), b, None) for i, b in enumerate(boxes)]


@pytest.mark.parametrize("true_angle", [0, 5, 12, 30, 60, 90, 135, 170])
def test_bar_direction_accuracy(true_angle):
    img, objs = frame_with([true_angle])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    direction, peak = bar_direction(gray, *knob_circle(objs[0]))
    assert direction is not None and peak > 2.5
    err = (direction - true_angle + 90) % 180 - 90
    assert abs(err) <= 3


def test_knob_without_a_bar_abstains():
    img = np.full((480, 1280, 3), 70, np.uint8)
    box = draw_knob(img, 300, 240, 40, 0, bar=False)
    direction, peak = bar_direction(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), *knob_circle(SamObject(0, "a", box, None)))
    assert direction is None


def test_read_knobs_off_slight_on_and_out_of_frame():
    ref_img, ref_objs = frame_with([0, 0, 0, 0])
    refs = {f"knob_{i + 1}": d for i, (d, _) in enumerate(measure_knobs(ref_img, ref_objs))}

    now, objs = frame_with([2, 10, 85, 178])    # off, slight, on, off (178 ~ 0 for a bar)
    r = {x.item_id: x for x in read_knobs(now, 0, objs, refs).items}
    assert r["knob_1"].state == "off"
    assert r["knob_2"].state is None and r["knob_2"].reason_code is ReasonCode.LOW_EVIDENCE
    assert r["knob_3"].state == "on" and 80 <= abs(r["knob_3"].angle_deg) <= 90
    assert r["knob_4"].state == "off"

    r = {x.item_id: x for x in read_knobs(now, 0, objs[:2], refs).items}   # 2 of 4 seen, no edge clue
    assert all(x.reason_code is ReasonCode.OUT_OF_FRAME for x in r.values())


def obj(x0, x1, y0=100, y1=180):
    return SamObject(0, str(x0), (x0, y0, x1, y1), None)


def test_assign_knobs_edges():
    ids = ["k1", "k2", "k3", "k4"]
    four = [obj(100, 180), obj(300, 380), obj(800, 880), obj(1000, 1080)]
    assert list(assign_knobs(four, ids, 1280)) == ids
    two_left = [obj(2, 80), obj(200, 280)]               # cut off on the left
    assert list(assign_knobs(two_left, ids, 1280)) == ["k3", "k4"]
    two_right = [obj(900, 980), obj(1200, 1279)]         # cut off on the right
    assert list(assign_knobs(two_right, ids, 1280)) == ["k1", "k2"]
    assert assign_knobs([obj(500, 580)], ids, 1280) == {}   # no edge clue: refuse to guess
