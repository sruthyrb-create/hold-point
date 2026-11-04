"""Oven display check: LED shape reader on synthetic seven-segment displays, plus the full
check with a faked Muse Spark (no key, no cost)."""
import json

import cv2
import numpy as np
import pytest

from holdpoint import oven
from holdpoint.budget import Ledger, Price
from holdpoint.meta_client import MetaClient
from tests.test_api_layer import FakeOpenAI

SEGS = {  # a b c d e f g
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
    "7": "abc", "8": "abcdefg", "9": "abcdfg", "P": "abefg", "r": "eg", "E": "afged", "b": "fedcg",
}


def draw_display(text, h=40, colour=(120, 255, 60), bg=30, size=(480, 640), noise=6, seed=0):
    """Render text like '10:16' or '350' as green seven-segment LEDs on a dark panel."""
    img = np.full((*size, 3), bg, np.uint8)
    w, t = int(h * 0.55), max(2, h // 8)
    x, y = 140, size[0] // 2 - h // 2
    for ch in text:
        if ch in ":;":                      # ";" = a colon blinked off (same slot, unlit)
            if ch == ":":
                for dy in (h // 3, 2 * h // 3):
                    cv2.rectangle(img, (x, y + dy - t // 2), (x + t, y + dy + t // 2), colour, -1)
            x += t + w // 2
            continue
        if ch != " ":
            p = {"a": ((x, y), (x + w, y + t)), "g": ((x, y + h // 2 - t // 2), (x + w, y + h // 2 + t // 2)),
                 "d": ((x, y + h - t), (x + w, y + h)), "f": ((x, y), (x + t, y + h // 2)),
                 "b": ((x + w - t, y), (x + w, y + h // 2)), "e": ((x, y + h // 2), (x + t, y + h)),
                 "c": ((x + w - t, y + h // 2), (x + w, y + h))}
            for s in SEGS[ch]:
                cv2.rectangle(img, p[s][0], p[s][1], colour, -1)
        x += w + w // 2
    n = np.random.default_rng(seed).normal(0, noise, img.shape)
    return (img + n).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("text,pattern", [("10:16", "clock"), ("9:05", "clock"), ("350", "cooking"),
                                          ("425", "cooking"), ("PrE", "cooking"), ("bE", "cooking")])
def test_led_shapes(text, pattern):
    r = oven.read_led(draw_display(text))
    assert r["pattern"] == pattern, r


def test_blank_and_far():
    assert oven.read_led(draw_display(""))["pattern"] is None
    far = oven.read_led(draw_display("350", h=7))
    assert far["pattern"] is None and far["reason"].value == "TOO_FAR"


def test_white_glare_is_not_a_segment():
    img = draw_display("10:16")
    cv2.circle(img, (450, 120), 40, (255, 255, 255), -1)          # white glare blob
    assert oven.read_led(img)["pattern"] == "clock"


def test_blinking_colon_counts_as_clock():
    frames = [draw_display("10:16"), draw_display("10;16"), draw_display("10:16"), draw_display("10;16")]
    geo, _ = oven.read_display_frames(frames)
    assert [g.items[0].state for g in geo] == ["off"] * 4


def spark(shows, text):
    return {"frames": [{"frame": i, "shows": shows, "text": text, "confidence": 0.9} for i in range(4)]}


def client(tmp_path, reply):
    ledger = Ledger(cap_usd=1.0, ledger_file=tmp_path / "l.jsonl", prices={"muse-spark-1.3": Price(1.25, 4.25)})
    return MetaClient(ledger=ledger, client=FakeOpenAI(reply), cache_dir=tmp_path / "c")


def test_oven_off_when_both_agree(tmp_path):
    r = oven.run_oven_check(client(tmp_path, spark("clock", "10:16")), [draw_display("10:16", seed=i) for i in range(4)])
    assert r["all_clear"] and r["answer"].startswith("Yes, the oven is off")


def test_oven_on_when_both_agree(tmp_path):
    r = oven.run_oven_check(client(tmp_path, spark("cooking", "350")), [draw_display("350", seed=i) for i in range(4)])
    assert r["danger"] and r["answer"] == "No. The oven is ON. Turn it off."


def test_spark_says_clock_but_display_shows_350(tmp_path):
    r = oven.run_oven_check(client(tmp_path, spark("clock", "3:50")), [draw_display("350", seed=i) for i in range(4)])
    assert not r["all_clear"] and r["danger"] and "may be ON" in r["answer"]


def test_timer_is_not_proof_of_off(tmp_path):
    r = oven.run_oven_check(client(tmp_path, spark("timer", "9:59")), [draw_display("9:59", seed=i) for i in range(4)])
    assert not r["all_clear"]


def test_blank_display_abstains_with_fix(tmp_path):
    r = oven.run_oven_check(client(tmp_path, spark("blank", None)), [draw_display("", seed=i) for i in range(4)])
    assert not r["all_clear"] and not r["danger"] and r["answer"].startswith("Can't confirm the oven.")


def test_server_oven_endpoint_and_agent(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from holdpoint import server
    c = client(tmp_path, spark("clock", "10:16"))
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "p")
    monkeypatch.setattr(server, "_client", lambda: c)
    monkeypatch.setattr(server, "save_frames", lambda *a: None)
    monkeypatch.setattr(server, "log_check", lambda r: "logged")
    http, h = TestClient(server.app), {"X-Holdpoint-Passcode": "p"}
    files = [("frames", (f"f{i}.jpg", cv2.imencode(".jpg", draw_display("10:16", seed=i))[1].tobytes(), "image/jpeg"))
             for i in range(4)]
    r = http.post("/api/check?item=oven&truth=oven_off", files=files, headers=h)
    assert r.status_code == 200 and r.json()["all_clear"], r.text
    assert http.post("/api/check?item=toaster", files=files, headers=h).status_code == 400
    assert "oven display" in server.CHECKABLE
