import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lastseen.eval_metrics import Outcome, classify, summarise
from lastseen import spatial, voice
from lastseen.answer import age_phrase, answer_where
from lastseen.home import Home, Item, Landmark
from lastseen.memory import CONFIRMED, PLACE_UNVERIFIED, MemoryStore, Sighting
from lastseen.sam import SamObject, parse_sam_text
from lastseen.watch import Pipeline, WatchState
from lastseen.witness_spark import Candidate, Match, parse_matches

FIX = Path(__file__).parent / "fixtures"


def make_home():
    return Home(
        landmarks=[Landmark("kitchen_stove", "stove", "kitchen", "stove"),
                   Landmark("living_room_sofa", "sofa", "living room", "sofa")],
        items=[Item("glasses", "glasses", "eyeglasses", "black reading glasses", ["specs"]),
               Item("pen", "pen", "pen", "blue pen")])


# ------------------------------------------------ spatial
def test_relations():
    lm = (100, 100, 200, 200)
    assert spatial.relation((120, 120, 140, 140), lm) == "on"
    assert spatial.relation((60, 140, 90, 160), lm) == "left_of"
    assert spatial.relation((210, 140, 240, 160), lm) == "right_of"
    assert spatial.relation((600, 140, 620, 160), lm) is None       # too far to anchor
    assert spatial.relation((150, 205, 170, 215), lm) == "near"      # just below, overlapping in x


def test_relations_agree():
    assert spatial.relations_agree("left_of", "left_of")
    assert spatial.relations_agree("near", "on")
    assert not spatial.relations_agree("left_of", "right_of")
    assert not spatial.relations_agree(None, "on")


def test_direction_in_frame():
    assert spatial.direction_in_frame((600, 300, 700, 400), 1280, 720).startswith("straight ahead")
    assert spatial.direction_in_frame((10, 300, 60, 340), 1280, 720).startswith("to your left")


# ------------------------------------------------ memory
def test_memory_retention_and_forget(tmp_path):
    st = MemoryStore(tmp_path / "m.jsonl", retention_hours=1)
    snap = tmp_path / "s.jpg"; snap.write_bytes(b"x")
    st.add(Sighting("glasses", 0, CONFIRMED, snapshot=str(snap)))
    st.add(Sighting("glasses", 5000, CONFIRMED))
    st.add(Sighting("pen", 5000, CONFIRMED))
    assert len(st.all(now=5000)) == 2          # the old one expired...
    assert not snap.exists()                   # ...and its snapshot was deleted
    assert st.forget("pen", now=5000) == 1
    assert [s.item for s in st.all(now=5000)] == ["glasses"]


# ------------------------------------------------ answers
def S(t, lm="kitchen_stove", rel="left_of", status=CONFIRMED, item="glasses"):
    return Sighting(item, t, status, lm, rel)


def test_never_seen():
    a = answer_where("glasses", [], make_home(), now=100)
    assert a["status"] == "never_seen" and "won't guess" in a["text"]


def test_confirmed_with_age():
    a = answer_where("glasses", [S(0)], make_home(), now=12 * 60)
    assert a["text"] == "Your glasses: left of the stove, in the kitchen, 12 minutes ago."
    assert a["status"] == "confirmed"


def test_stale_says_may_have_moved():
    a = answer_where("glasses", [S(0)], make_home(), now=45 * 60)
    assert a["status"] == "stale" and "may have moved" in a["text"]


def test_newer_unverified_place_is_unsure_and_gives_last_confirmed():
    s = [S(0), S(300, lm="living_room_sofa", rel="on", status=PLACE_UNVERIFIED)]
    a = answer_where("glasses", s, make_home(), now=360)
    assert a["status"] == "unsure"
    assert "possibly on the sofa" in a["text"] and "Before that" in a["text"] and "left of the stove" in a["text"]


def test_unverified_at_same_place_supports_confirmed():
    s = [S(0), S(300, status=PLACE_UNVERIFIED)]
    assert answer_where("glasses", s, make_home(), now=360)["status"] == "confirmed"


def test_two_places():
    s = [S(0, lm="living_room_sofa", rel="on"), S(60)]
    a = answer_where("glasses", s, make_home(), now=120)
    assert a["status"] == "two_places" and "sofa" in a["text"] and "stove" in a["text"]


def test_age_phrase():
    assert age_phrase(10) == "just now" and age_phrase(60) == "1 minute ago" and age_phrase(3 * 3600) == "about 3 hours ago"


# ------------------------------------------------ voice intents
def test_intents():
    names = make_home().item_names()
    assert voice.parse_intent("Where are my specs?", names) == ("where", "glasses")
    assert voice.parse_intent("have you seen my pen", names) == ("where", "pen")
    assert voice.parse_intent("where did I leave it", names) == ("where", None)
    assert voice.parse_intent("stop watching", names) == ("watch_off", None)
    assert voice.parse_intent("start watching", names) == ("watch_on", None)
    assert voice.parse_intent("forget everything", names) == ("forget", None)
    assert voice.parse_intent("what a nice day", names) == (None, None)


# ------------------------------------------------ Spark parsing
def test_parse_matches_drops_invented_and_malformed():
    raw = {"candidates": [
        {"candidate": 0, "is_users_item": True, "confidence": 1.4, "landmark": "kitchen_stove",
         "relation": "left_of", "reason": "same frame"},
        {"candidate": 0, "is_users_item": False, "confidence": 0.1, "landmark": "none", "relation": "none", "reason": ""},
        {"candidate": 7, "is_users_item": True, "confidence": 0.9, "landmark": "none", "relation": "none", "reason": ""},
        {"candidate": 1, "is_users_item": True, "confidence": 0.9, "landmark": "garage", "relation": "on", "reason": ""},
        {"candidate": 2},
    ]}
    ms = parse_matches(raw, 3, {"kitchen_stove", "living_room_sofa"})
    assert [m.candidate for m in ms] == [0, 1]
    assert ms[0].confidence == 1.0 and ms[0].landmark == "kitchen_stove"
    assert ms[1].landmark is None and ms[1].relation is None      # invented landmark dropped


def test_real_sam_output_parses():
    objs = parse_sam_text((FIX / "sam_four_knobs.txt").read_text())
    assert len(objs) == 4 and all(o.mask is not None for o in objs)


# ------------------------------------------------ watching pipeline (fake witnesses)
def scene():
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, (720, 1280, 3), dtype=np.uint8)   # bright, textured -> passes quality


def fake_pipeline(tmp_path, item_boxes, lm_boxes, matches):
    def segment(c, img, phrase, tag=""):
        boxes = item_boxes.get(phrase, []) + lm_boxes.get(phrase, [])
        return [SamObject(0, str(i), b, None, 0.9) for i, b in enumerate(boxes)]

    def confirm(c, frame, cands, home, tag=""):
        return [Match(i, *matches[i]) for i in range(len(cands)) if i in matches]

    return Pipeline(None, make_home(), MemoryStore(tmp_path / "m.jsonl"), segment=segment,
                    confirm=confirm, keep_snapshots=False)


def test_pipeline_confirms_place_when_witnesses_agree(tmp_path):
    p = fake_pipeline(tmp_path, {"eyeglasses": [(500, 400, 560, 430)]}, {"stove": [(600, 300, 900, 600)]},
                      {0: (True, 0.9, "kitchen_stove", "left_of", "match")})
    r = p.process(scene(), now=1000)
    assert r["processed"] and r["sightings"][0]["status"] == CONFIRMED
    assert p.store.all(now=1000)[0].relation == "left_of"


def test_pipeline_place_unverified_when_geometry_disagrees(tmp_path):
    p = fake_pipeline(tmp_path, {"eyeglasses": [(500, 400, 560, 430)]}, {"stove": [(600, 300, 900, 600)]},
                      {0: (True, 0.9, "kitchen_stove", "right_of", "match")})
    r = p.process(scene(), now=1000)
    assert r["sightings"][0]["status"] == PLACE_UNVERIFIED


def test_pipeline_stores_nothing_when_spark_rejects(tmp_path):
    p = fake_pipeline(tmp_path, {"eyeglasses": [(500, 400, 560, 430)]}, {},
                      {0: (False, 0.9, None, None, "different glasses")})
    p.process(scene(), now=1000)
    assert p.store.all(now=1000) == []


def test_pipeline_abstains_on_tie(tmp_path):
    p = fake_pipeline(tmp_path, {"eyeglasses": [(100, 100, 150, 130), (800, 100, 850, 130)]}, {},
                      {0: (True, 0.9, None, None, ""), 1: (True, 0.85, None, None, "")})
    r = p.process(scene(), now=1000)
    assert r["abstained"] == ["glasses"] and p.store.all(now=1000) == []


def test_pipeline_skips_dark_and_unchanged_frames(tmp_path):
    p = fake_pipeline(tmp_path, {}, {}, {})
    assert p.process(np.zeros((720, 1280, 3), np.uint8), now=0)["skipped"] == "too_dark"
    f = scene()
    assert p.process(f, now=10)["processed"]
    assert p.process(f, now=20)["skipped"] == "unchanged"
    assert p.process(f, now=80)["processed"]      # re-checked after a minute anyway


def test_find(tmp_path):
    p = fake_pipeline(tmp_path, {"eyeglasses": [(600, 300, 700, 400)]}, {}, {0: (True, 0.9, None, None, "")})
    r = p.find(scene(), "glasses")
    assert r["found"] and "straight ahead" in r["text"]


# ------------------------------------------------ enrolment (fake SAM + fake Spark)
class FakeClient:
    def chat_json(self, *a, **k):
        return {"description": "black glasses with thin frame"}


def test_enrol_item_and_landmark(tmp_path, monkeypatch):
    from lastseen.enrol import EnrolError, add_landmark, enrol_item
    monkeypatch.chdir(tmp_path)
    home = Home()
    seg = lambda c, img, p: [SamObject(0, "0", (100, 100, 200, 160), None, 0.8)]
    item = enrol_item(FakeClient(), home, [scene(), scene(), scene()], "Glasses", "eyeglasses",
                      segment=seg, items_dir=tmp_path / "items")
    assert item.id == "glasses" and len(item.references) == 3 and item.description.startswith("black")
    assert Home.load().item("glasses").sam_phrase == "eyeglasses"
    with pytest.raises(EnrolError):
        enrol_item(FakeClient(), home, [scene(), scene()], "keys", segment=lambda c, i, p: [],
                   items_dir=tmp_path / "items")
    lm = add_landmark(None, home, scene(), "Stove", "Kitchen", segment=seg)
    assert lm.id == "kitchen_stove" and Home.load().landmark("kitchen_stove").room == "kitchen"


# ------------------------------------------------ eval metrics
def test_metrics():
    o = [Outcome("a", ["x"], "confirmed", "x"), Outcome("b", ["y"], "confirmed", "x"),
         Outcome("c", [], "unsure", "x"), Outcome("d", ["y", "x"], "two_places", "x")]
    assert [classify(i) for i in o] == ["correct", "wrong", "not_sure", "correct"]
    s = summarise(o)
    assert s["confident_wrong_rate"] == 0.25 and s["precision_when_committed"] == round(2 / 3, 3)


# ------------------------------------------------ server (no model calls)
def test_server_ask_and_passcode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LASTSEEN_PASSCODE", "pw")
    make_home().save()
    MemoryStore().add(Sighting("glasses", __import__("time").time() - 120, CONFIRMED, "kitchen_stove", "on"))
    from lastseen import server
    c = TestClient(server.app)
    assert c.post("/api/ask", json={"text": "where are my glasses"}).status_code == 401
    r = c.post("/api/ask", json={"text": "where are my glasses"}, headers={"X-LastSeen-Passcode": "pw"}).json()
    assert r["status"] == "confirmed" and "on the stove" in r["text"]
    r = c.post("/api/ask", json={"text": "where did I put it"}, headers={"X-LastSeen-Passcode": "pw"}).json()
    assert r["status"] == "which"
    assert "Last Seen" in c.get("/").text


def test_mask_stops_false_on():
    lm = (100, 100, 300, 300)
    mask = np.zeros((200, 200), bool); mask[:, 80:] = True       # stove body is the right part of its box
    item = (110, 200, 150, 230)                                   # inside the box, but left of the stove itself
    assert spatial.relation(item, lm) == "on"                     # box-only geometry over-claims
    assert spatial.relation(item, lm, mask) == "left_of"
    assert spatial.relation((250, 150, 280, 180), lm, mask) == "on"


def test_unsure_uses_spark_surface_without_landmark():
    s = [S(0), Sighting("glasses", 300, PLACE_UNVERIFIED, seen_on="wooden dining table")]
    a = answer_where("glasses", s, make_home(), now=360)
    assert a["status"] == "unsure" and "looked like a wooden dining table" in a["text"]
    assert "no landmark I know" in a["text"] and "Before that" in a["text"]


def test_surface_parsed_and_unknown_dropped():
    raw = {"candidates": [
        {"candidate": 0, "is_users_item": True, "confidence": 0.9, "landmark": "none", "relation": "none",
         "reason": "", "surface": "Wooden Dining Table."},
        {"candidate": 1, "is_users_item": True, "confidence": 0.9, "landmark": "none", "relation": "none",
         "reason": "", "surface": "unknown"}]}
    ms = parse_matches(raw, 2, set())
    assert ms[0].surface == "wooden dining table" and ms[1].surface is None
