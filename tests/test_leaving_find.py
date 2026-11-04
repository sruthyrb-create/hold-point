"""Optional last step of the leaving-home check: find the user's keys with Last Seen."""
import json
import time

import cv2
import numpy as np

from holdpoint import agent
from holdpoint.server import leaving_find
from lastseen.find_check import run_find_check
from lastseen.home import Home, Item, Landmark
from lastseen.memory import CONFIRMED, MemoryStore, Sighting
from tests.test_agent import OFF, ON, call, client
from tests.test_lastseen_core import fake_pipeline, scene

KEYS_WHERE = "Your keys: left of the stove, in the kitchen, 20 minutes ago."
FOUND = {"kind": "find", "found": True, "all_clear": False, "danger": False,
         "answer": "Found your keys: straight ahead, close.", "guidance": None}
MISSING = {"kind": "find", "found": False, "all_clear": False, "danger": False,
           "answer": "I can't see your keys here. Turn slowly.", "guidance": "Turn slowly."}


def ls_home():
    return Home(landmarks=[Landmark("kitchen_stove", "stove", "kitchen", "stove")],
                items=[Item("keys", "keys", "keys", "silver house keys on a red ring")])


# ---------------------------------------------------------------- the switch
def test_switch_off_by_default_and_only_enrolled_items(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert leaving_find({}) == {}
    assert leaving_find({"HOLDPOINT_LEAVING_FIND": "keys"}) == {}          # nothing enrolled yet
    ls_home().save()
    MemoryStore().add(Sighting("keys", time.time() - 20 * 60, CONFIRMED, "kitchen_stove", "left_of"))
    got = leaving_find({"HOLDPOINT_LEAVING_FIND": "keys, wallet"})
    assert list(got) == ["keys"] and got["keys"] == KEYS_WHERE


# ---------------------------------------------------------------- the agent
def test_belongings_checked_last_and_told_to_wonder(tmp_path):
    c = client(tmp_path, [call("check_item", item="stove", instruction="Look at the stove knobs."),
                          call("check_item", item="keys", instruction="Now your keys, left of the stove."),
                          call("finish")])
    s, step = agent.start(c, ["stove"], {"stove"}, "I'm leaving", find_where={"keys": KEYS_WHERE})
    assert s.checklist == ["stove", "keys"] and step.item == "stove"
    first = json.loads(c.client.sent[0][1]["content"])
    assert first["items_to_find"] == {"keys": KEYS_WHERE}
    step = agent.submit(c, s, OFF)
    assert step.item == "keys"
    tool = json.loads(c.client.sent[-1][-1]["content"])
    assert tool["verified_off"] is True                                      # stove result unchanged
    step = agent.submit(c, s, FOUND)
    assert step.action == "done" and step.all_clear
    assert step.summary.startswith("All clear. Stove verified off. Keys found. Checked")


def test_missing_keys_do_not_spoil_all_clear_but_are_reported(tmp_path):
    c = client(tmp_path, [None])                                             # Wonder unusable -> fixed plan
    s, step = agent.start(c, ["stove"], {"stove"}, find_where={"keys": KEYS_WHERE})
    assert s.planner == "fixed" and step.item == "stove"
    step = agent.submit(c, s, OFF)
    assert step.item == "keys" and "left of the stove" in step.say
    for _ in range(agent.MAX_ATTEMPTS):
        step = agent.submit(c, s, MISSING)
    assert step.action == "done" and step.all_clear
    assert "Stove verified off. Couldn't find your keys. " + KEYS_WHERE in step.summary


def test_stove_on_stays_not_all_clear_even_with_keys_found(tmp_path):
    c = client(tmp_path, [None])
    s, step = agent.start(c, ["stove"], {"stove"}, find_where={"keys": KEYS_WHERE})
    for _ in range(agent.MAX_ATTEMPTS):
        step = agent.submit(c, s, ON)
    step = agent.submit(c, s, FOUND)
    assert step.action == "done" and not step.all_clear
    assert step.summary.startswith("Not all clear. The stove may still be on. Keys found.")


def test_wonder_never_gets_a_found_without_verification(tmp_path):
    c = client(tmp_path, [call("check_item", item="keys", instruction="Your keys are on the table."),
                          call("finish")])
    s, step = agent.start(c, [], {"stove"}, find_where={"keys": KEYS_WHERE})
    step = agent.submit(c, s, MISSING)
    tool = json.loads(c.client.sent[-1][-1]["content"])
    assert tool == {"item": "keys", "verified_found": False, "verified_answer": MISSING["answer"],
                    "guidance": "Turn slowly.", "attempts_used": 1, "attempts_left": agent.MAX_ATTEMPTS - 1}
    assert "Couldn't find your keys" in step.summary


# ---------------------------------------------------------------- the find check
class Ledgerish:
    def __init__(self): self.n = 0.0
    def spent(self): return self.n


def test_find_check_stops_at_first_verified_view(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = ls_home()
    p = fake_pipeline(tmp_path, {"keys": [(600, 300, 700, 400)]}, {}, {0: (True, 0.9, None, None, "")})
    p.home = home
    seen = []
    orig = p.find
    p.find = lambda img, item, lm=None: (seen.append(lm), orig(img, item, lm))[1]
    c = type("C", (), {"ledger": Ledgerish()})()
    r = run_find_check(c, [scene(), scene(), scene()], "keys", home, MemoryStore(), p)
    assert r["kind"] == "find" and r["found"] and len(seen) == 1 and r["answer"].startswith("Found your keys")
    assert [x["model"] for x in r["calls"]] == ["sam-3.1", "muse-spark-1.3"]


def test_find_check_not_found_guides_to_last_place(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = ls_home()
    store = MemoryStore()
    store.add(Sighting("keys", time.time() - 60, CONFIRMED, "kitchen_stove", "left_of"))
    p = fake_pipeline(tmp_path, {}, {"stove": [(600, 300, 900, 600)]}, {})
    p.home = home
    c = type("C", (), {"ledger": Ledgerish()})()
    r = run_find_check(c, [scene(), scene()], "keys", home, store, p)
    assert not r["found"] and r["guidance"].startswith("I can see the stove")
    p2 = fake_pipeline(tmp_path, {}, {}, {})
    p2.home = home
    r2 = run_find_check(c, [scene()], "keys", home, store, p2)
    assert r2["guidance"] == "Last seen near the stove, in the kitchen. Go there and turn slowly."


# ---------------------------------------------------------------- through the server
def test_agent_endpoints_find_keys(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from holdpoint import server
    import lastseen.find_check as fc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "pw")
    monkeypatch.setenv("HOLDPOINT_LEAVING_FIND", "keys")
    ls_home().save()
    MemoryStore().add(Sighting("keys", time.time() - 300, CONFIRMED, "kitchen_stove", "left_of"))
    monkeypatch.setattr(server, "leaving_checklist", lambda *a, **k: [])        # keys only, no stove setup needed
    c = client(tmp_path, [None])
    monkeypatch.setitem(server._state, "client", c)
    fake = fake_pipeline(tmp_path, {"keys": [(600, 300, 700, 400)]}, {}, {0: (True, 0.9, None, None, "")})
    fake.home = ls_home()
    monkeypatch.setattr(fc, "Pipeline", lambda *a, **k: fake)
    h = {"X-Holdpoint-Passcode": "pw"}
    tc = TestClient(server.app)
    r = tc.post("/api/agent/start", json={"said": "I'm leaving"}, headers=h).json()
    assert r["action"] == "check" and r["item"] == "keys" and r["kind"] == "find" and "left of the stove" in r["say"]
    jpg = cv2.imencode(".jpg", scene())[1].tobytes()
    r2 = tc.post(f"/api/agent/step?session={r['session']}", files={"frames": ("f.jpg", jpg, "image/jpeg")}, headers=h).json()
    assert r2["check"]["found"] and r2["action"] == "done" and "Keys found." in r2["say"]
    assert not (tmp_path / "home" / "live").exists()                          # find frames not saved as eval data


# ---------------------------------------------------------------- reminder + "where are my keys?" (current flow)
def test_reminder_only_when_all_clear(tmp_path):
    c = client(tmp_path, [None])
    s, step = agent.start(c, ["stove"], {"stove"}, reminder=["keys", "glasses", "wallet"])
    step = agent.submit(c, s, ON)                                   # ON -> turn it off and re-check
    assert step.action == "check" and "turn the stove off" in step.say.lower()
    step = agent.submit(c, s, OFF)
    assert step.action == "done" and step.all_clear
    assert ("All clear. Stove verified off. You're all set. Don't forget your keys, glasses and wallet. Checked"
            in step.summary)
    c2 = client(tmp_path / "b", [None])
    s2, step2 = agent.start(c2, ["stove"], {"stove"}, reminder=["keys"])
    for _ in range(agent.MAX_ATTEMPTS):
        step2 = agent.submit(c2, s2, ON)
    assert not step2.all_clear and "Don't forget" not in step2.summary


def three_items():
    return Home(landmarks=[Landmark("kitchen_stove", "stove", "kitchen", "stove"),
                           Landmark("study_computer_table", "computer table", "study", "desk"),
                           Landmark("living_room_couch", "couch", "living room", "couch")],
                items=[Item("keys", "keys", "keys", "silver keys"), Item("glasses", "glasses", "eyeglasses", "gold glasses"),
                       Item("wallet", "wallet", "wallet", "brown leather wallet")])


def test_item_questions_win_over_leaving(tmp_path, monkeypatch):
    from holdpoint.server import lastseen_question, leaving_reminder
    monkeypatch.chdir(tmp_path)
    assert lastseen_question("where are my keys") is None                    # nothing enrolled: Hold Point as before
    three_items().save()
    assert lastseen_question("where did I leave my keys?") == ("where", "keys")
    assert lastseen_question("Where's my key") == ("where", "keys")
    assert lastseen_question("guide me to my wallet") == ("guide", "wallet")
    assert lastseen_question("I'm leaving, check the house") is None
    assert leaving_reminder() == ["keys", "glasses", "wallet"]


def test_voice_endpoint_answers_where_from_memory(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from holdpoint import server
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "pw")
    three_items().save()
    MemoryStore().add(Sighting("keys", time.time() - 300, CONFIRMED, "study_computer_table", "on"))
    MemoryStore().add(Sighting("wallet", time.time() - 120, "place_unverified", seen_on="brown wooden desk"))
    monkeypatch.setitem(server._state, "client", client(tmp_path, []))
    said = iter(["where did I leave my keys", "where is my wallet", "where are my glasses", "I'm leaving"])
    monkeypatch.setattr(server, "transcribe", lambda c, d: {"transcript": next(said), "seconds": 1.0})
    tc = TestClient(server.app)
    post = lambda: tc.post("/api/voice", files={"audio": ("a.wav", b"x", "audio/wav")},
                           headers={"X-Holdpoint-Passcode": "pw"}).json()
    r = post()
    assert r["action"] == "where" and r["reply"].startswith("Your keys: on the computer table, in the study, 5 minutes ago.")
    assert "guide me to my keys" in r["reply"]
    r = post()
    assert "looked like a brown wooden desk" in r["reply"] and r["status"] == "unsure"
    r = post()
    assert r["reply"] == "I haven't seen your glasses yet, so I won't guess."
    assert post()["intent"] == "leaving"


def test_watch_frame_uses_sharpest(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from lastseen import server as ls
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "pw")
    monkeypatch.setenv("LASTSEEN_SAVE_FRAMES", "0")
    got = []

    class P:
        state = None
        def process(self, img):
            got.append(img)
            return {"processed": True, "skipped": None, "candidates": [], "sightings": [], "calls": []}
    monkeypatch.setitem(ls._state, "pipeline", P())
    monkeypatch.setitem(ls._state, "watching", True)
    monkeypatch.setitem(ls._state, "client", client(tmp_path, []))
    sharp = scene()
    blurry = cv2.GaussianBlur(sharp, (31, 31), 10)
    enc = lambda im: cv2.imencode(".jpg", im)[1].tobytes()
    r = TestClient(ls.app).post("/api/frame", headers={"X-LastSeen-Passcode": "pw"},
                                files=[("frames", ("a.jpg", enc(blurry), "image/jpeg")),
                                       ("frames", ("b.jpg", enc(sharp), "image/jpeg")),
                                       ("frames", ("c.jpg", enc(blurry), "image/jpeg"))]).json()
    assert len(r["sharpness"]) == 3 and r["sharpness"].index(max(r["sharpness"])) == 1
    assert round(ls.sharpness(got[0]), 1) == max(r["sharpness"])
