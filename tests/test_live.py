"""Live check pipeline and server, with fake models (no key, no cost)."""
import json

import cv2
import numpy as np
import pytest

from holdpoint.budget import Ledger, Price
from holdpoint.check import UNAVAILABLE, run_stove_check
from holdpoint.meta_client import MetaClient
from holdpoint.witness_sam import measure_knobs
from tests.test_api_layer import FakeOpenAI
from tests.test_witness_sam import frame_with

IDS = ["knob_1", "knob_2", "knob_3", "knob_4"]


def textured(img, seed=0):
    """Add camera-like grain so the frame passes the blur check like a real photo."""
    noise = np.random.default_rng(seed).normal(0, 12, img.shape)
    return (img + noise).clip(0, 255).astype(np.uint8)


def spark_reply(states, n_frames=4, conf=0.9):
    return {"frames": [{"frame": f, "items": [
        {"item_id": i, "visible": True, "state": s, "confidence": conf, "reason_code": "OK"}
        for i, s in zip(IDS, states)]} for f in range(n_frames)]}


def setup(tmp_path, true_angles, spark_states, fail_sam=False):
    ref_img, ref_objs = frame_with([0, 0, 0, 0])
    refs = {k: d for k, (d, _) in zip(IDS, measure_knobs(ref_img, ref_objs))}
    stove = {"sam_prompt": "stove knob", "knobs": [
        {"id": k, "name": f"knob {n}", "position": "left" if n <= 2 else "right", "description": k}
        for n, k in enumerate(IDS, 1)]}
    img, objs = frame_with(true_angles)
    frames = [textured(img, s) for s in range(4)]

    def fake_segment(client, image, prompt, tag=""):
        if fail_sam:
            raise TimeoutError("SAM timed out")
        client.ledger.record_units("sam-3.1", images=1, tag=tag)
        return objs

    ledger = Ledger(cap_usd=1.0, ledger_file=tmp_path / "ledger.jsonl",
                    prices={"muse-spark-1.3": Price(1.25, 4.25), "sam-3.1": Price(per_image=0.0025)})
    client = MetaClient(ledger=ledger, client=FakeOpenAI(spark_reply(spark_states)), cache_dir=tmp_path / "cache")
    return client, frames, stove, refs, fake_segment


def test_frames_pass_quality_checks(tmp_path):
    from holdpoint.quality import frame_reasons
    _, frames, *_ = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4)
    assert frame_reasons(frames[0]) == []


def test_all_off_both_witnesses_agree(tmp_path):
    client, frames, stove, refs, seg = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4)
    r = run_stove_check(client, frames, stove, refs, segment=seg)
    assert r["answer"] == "Yes, all four off."
    assert r["all_clear"] and not r["danger"]
    assert r["cost_usd"] == pytest.approx(4 * 0.0025 + (2000 * 1.25 + 300 * 4.25) / 1e6, abs=1e-4)
    assert {c["model"] for c in r["calls"]} == {"sam-3.1", "muse-spark-1.3"}
    assert r["overlay"].startswith("data:image/jpeg;base64,")


def test_spark_says_all_on_but_geometry_finds_only_knob_3(tmp_path):
    # The real failure from 31 Oct: Spark called all four ON. Hold Point must not repeat it.
    client, frames, stove, refs, seg = setup(tmp_path, [0, 0, 85, 0], ["on"] * 4)
    r = run_stove_check(client, frames, stove, refs, segment=seg)
    assert r["answer"].startswith("No. knob 3 is ON")
    by = {it["id"]: it["verdict"] for it in r["items"]}
    assert by["knob_1"]["answered"] is False and by["knob_1"]["reason"] == "WITNESS_DISAGREE"
    assert r["danger"] and not r["all_clear"]


def test_failed_model_call_never_becomes_all_clear(tmp_path):
    client, frames, stove, refs, seg = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4, fail_sam=True)
    r = run_stove_check(client, frames, stove, refs, segment=seg)
    assert r["answer"] == UNAVAILABLE and not r["all_clear"] and r["errors"]


# ---------------------------------------------------------------- server

@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from holdpoint import server

    client, frames, stove, refs, seg = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4)
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "secret")
    monkeypatch.setattr(server, "_stove", lambda: (stove, refs))
    monkeypatch.setattr(server, "_client", lambda: client)
    monkeypatch.setattr(server, "log_check", lambda result: "Checked 9:02 PM: " + result["answer"])
    monkeypatch.setattr(server, "save_frames", lambda images, result: None)
    real = server.run_stove_check
    monkeypatch.setattr(server, "run_stove_check", lambda *a, **k: real(*a, segment=seg, **k))
    files = [("frames", (f"f{i}.jpg", cv2.imencode(".jpg", f)[1].tobytes(), "image/jpeg"))
             for i, f in enumerate(frames)]
    return TestClient(server.app), files


def test_server_needs_passcode(api):
    http, files = api
    assert http.post("/api/check", files=files).status_code == 401
    assert http.post("/api/check", files=files, headers={"X-Holdpoint-Passcode": "nope"}).status_code == 401


def test_server_check_end_to_end(api):
    http, files = api
    res = http.post("/api/check", files=files, headers={"X-Holdpoint-Passcode": "secret"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["answer"] == "Yes, all four off." and body["record"].startswith("Checked")
    assert http.get("/").status_code == 200


def test_log_check_writes_record(tmp_path):
    from holdpoint.server import log_check
    sentence = log_check({"answer": "Yes, all four off.", "all_clear": True, "cost_usd": 0.02}, home=tmp_path)
    line = json.loads((tmp_path / "check_log.jsonl").read_text())
    assert line["all_clear"] and sentence.endswith("Yes, all four off.")


# ---------------------------------------------------------------- frame consensus

from holdpoint.check import geometry_consensus  # noqa: E402
from holdpoint.schemas import FrameReadings, ItemReading, ReasonCode  # noqa: E402


def geo_frames(turns):
    """One knob's per-frame turn from OFF, as read_knobs would report it."""
    from holdpoint.geometry import state_from_angle
    out = []
    for i, t in enumerate(turns):
        r = state_from_angle("k", t, 0.0, period=180.0).model_copy(update={"angle_deg": t})
        out.append(FrameReadings(frame_index=i, witness="geometry", items=[r]))
    return out


def states(frames):
    return [f.items[0].state for f in frames]


def test_consensus_rescues_one_edge_frame():
    raw = geo_frames([-5.3, -4.5, -7.0, -3.3])          # frame 3 alone is in the unsure band
    assert states(raw) == ["off", "off", None, "off"]
    assert states(geometry_consensus(raw, ["k"])) == ["off"] * 4


def test_consensus_keeps_a_real_turn_on():
    assert states(geometry_consensus(geo_frames([80, 82, 79, 81]), ["k"])) == ["on"] * 4


def test_inconsistent_frames_are_left_alone():
    raw = geo_frames([0.5, -1.0, 40.0, 0.0])            # one frame saw a big turn
    assert states(geometry_consensus(raw, ["k"])) == states(raw)


def test_consensus_handles_the_180_wrap():
    assert states(geometry_consensus(geo_frames([88, -89, 89, -88]), ["k"])) == ["on"] * 4


def test_slight_turn_stays_unsure():
    assert states(geometry_consensus(geo_frames([12, 13, 11, 14]), ["k"])) == [None] * 4


# ---------------------------------------------------------------- setup from the phone

def test_setup_from_frames_saves_off_directions(tmp_path):
    from holdpoint.profile import load_stove
    from holdpoint.setup_live import setup_from_frames
    client, frames, _, _, seg = setup(tmp_path, [3, 3, 3, 3], ["off"] * 4)
    r = setup_from_frames(client, frames, 4, segment=seg, home=tmp_path)
    assert r["frames_used"] == 4
    stove, refs = load_stove(tmp_path)
    assert all(abs(((d - 3) + 90) % 180 - 90) <= 3 for d in refs.values())
    assert (tmp_path / "stove_setup.jpg").exists()


def test_setup_refuses_wrong_knob_count(tmp_path):
    from holdpoint.setup_live import SetupFailed, setup_from_frames
    client, frames, _, _, seg = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4)
    with pytest.raises(SetupFailed):
        setup_from_frames(client, frames, 3, segment=seg, home=tmp_path)


def test_spark_reads_knob_row_close_ups(tmp_path):
    import base64
    client, frames, stove, refs, seg = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4)
    tall = [np.vstack([f, np.full((600, f.shape[1], 3), 90, np.uint8)]) for f in frames]   # extra kitchen below
    run_stove_check(client, tall, stove, refs, segment=seg)
    sent = [c["image_url"]["url"] for c in client.client.last["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(sent) == 4
    img = cv2.imdecode(np.frombuffer(base64.b64decode(sent[0].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] < 0.5 * tall[0].shape[0] * img.shape[1] / tall[0].shape[1]   # a band, not the frame


def test_setup_tolerates_one_wobbly_frame(tmp_path):
    from holdpoint.setup_live import setup_from_frames
    from holdpoint.witness_sam import SamObject
    client, _, _, _, _ = setup(tmp_path, [0, 0, 0, 0], ["off"] * 4)
    angles = [[-2, 0, 0, 0], [4, 0, 0, 0], [6.5, 0, 0, 0], [1, 0, 0, 0], [14, 0, 0, 0]]  # knob 1: one outlier
    imgs = [textured(frame_with(a)[0], i) for i, a in enumerate(angles)]
    objs = frame_with([0, 0, 0, 0])[1]
    r = setup_from_frames(client, imgs, 4, segment=lambda c, im, p, tag="": objs, home=tmp_path)
    assert r["frames_used"] == 5 and abs(r["off_directions"][0] - 2.5) < 3
