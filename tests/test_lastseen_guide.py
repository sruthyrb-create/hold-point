import json
from types import SimpleNamespace as NS

from lastseen import guide, voice
from lastseen.memory import CONFIRMED, PLACE_UNVERIFIED, Sighting
from tests.test_lastseen_core import make_home, scene, fake_pipeline


class FakeLedger:
    def check(self, *a, **k): pass
    def record(self, *a, **k): pass


def wonder_client(script):
    """script: list of (tool_name, args) Wonder returns in order; 'TEXT' = no tool call."""
    calls = iter(script)

    def create(**kw):
        name, args = next(calls)
        tc = [] if name == "TEXT" else [NS(id=f"c{len(kw['messages'])}", function=NS(name=name, arguments=json.dumps(args)))]
        return NS(usage=NS(prompt_tokens=100, completion_tokens=20),
                  choices=[NS(message=NS(content="hi", tool_calls=tc))])
    return NS(ledger=FakeLedger(), client=NS(chat=NS(completions=NS(create=create))))


def sightings():
    return [Sighting("glasses", 0, CONFIRMED, "kitchen_stove", "left_of"),
            Sighting("glasses", 500, PLACE_UNVERIFIED, "living_room_sofa", "on"),
            Sighting("glasses", 400, CONFIRMED, "kitchen_stove", "on")]


def test_build_places_newest_first_one_per_landmark_ends_here():
    ps = guide.build_places("glasses", sightings(), make_home(), now=600)
    assert [p.id for p in ps] == ["living_room_sofa", "kitchen_stove", guide.HERE]
    assert ps[1].phrase == "on the stove, in the kitchen" and ps[1].age_s == 200


def test_newest_sighting_without_landmark_is_searched_first():
    ss = sightings() + [Sighting("glasses", 550, PLACE_UNVERIFIED, seen_on="wooden dining table")]
    ps = guide.build_places("glasses", ss, make_home(), now=600)
    assert ps[0].id == guide.SEEN and "wooden dining table" in guide.default_instruction(ps[0])


def test_wonder_guides_and_found_ends_without_another_wonder_call():
    c = wonder_client([("look_at", {"place": "kitchen_stove", "instruction": "Go to the kitchen, face the stove."})])
    s, step = guide.start(c, "glasses", make_home(), sightings(), now=600)
    assert step.action == "look" and step.place == "kitchen_stove" and "kitchen" in step.say   # Wonder may choose
    assert guide.add_view(c, s, {"found": False, "landmark_seen": True}) is None     # keep looking
    done = guide.add_view(c, s, {"found": True, "text": "Found your glasses: straight ahead, close."})
    assert done.action == "done" and done.found and done.say.startswith("Found your glasses")
    tool_msgs = [m for m in s.messages if m["role"] == "tool"]
    assert json.loads(tool_msgs[-1]["content"])["item_found"] is True


def test_wonder_cannot_pick_unknown_place_fixed_plan_takes_over():
    c = wonder_client([("look_at", {"place": "garage", "instruction": "Your glasses are in the garage."})])
    s, step = guide.start(c, "glasses", make_home(), sightings(), now=600)
    assert s.planner == "fixed" and step.place == "living_room_sofa" and "garage" not in step.say


def test_fixed_plan_runs_out_and_says_not_found():
    c = wonder_client([("TEXT", {})])
    s, step = guide.start(c, "glasses", make_home(), sightings(), now=600)
    n = 0
    while step.action == "look":
        for _ in range(guide.LOOK_VIEWS):
            r = guide.add_view(c, s, {"found": False, "landmark_seen": False})
        step = r; n += 1
    assert n == 3 * guide.MAX_ATTEMPTS
    assert not step.found and "couldn't find your glasses" in step.say and "may have moved" in step.say


def test_never_seen_still_looks_here():
    c = wonder_client([("TEXT", {})])
    s, step = guide.start(c, "glasses", make_home(), [], now=600)
    assert step.place == guide.HERE and "turn slowly" in step.say


def test_find_reports_landmark(tmp_path):
    p = fake_pipeline(tmp_path, {}, {"stove": [(600, 300, 900, 600)]}, {})
    r = p.find(scene(), "glasses", "kitchen_stove")
    assert r["found"] is False and r["landmark_seen"] is True


def test_guide_intent():
    names = make_home().item_names()
    assert voice.parse_intent("guide me to my glasses", names) == ("guide", "glasses")
    assert voice.parse_intent("help me find my pen", names) == ("guide", "pen")
    assert voice.parse_intent("take me to it", names) == ("guide", None)


def test_guide_endpoints_end_to_end(tmp_path, monkeypatch):
    import time
    import cv2
    from fastapi.testclient import TestClient
    from lastseen import server
    from lastseen.memory import MemoryStore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LASTSEEN_PASSCODE", "pw")
    make_home().save()
    MemoryStore().add(Sighting("glasses", time.time() - 120, CONFIRMED, "kitchen_stove", "left_of"))
    c = wonder_client([("look_at", {"place": "kitchen_stove", "instruction": "Go to the kitchen and face the stove."})])
    c.ledger.summary = lambda: "$0 spent"
    p = fake_pipeline(tmp_path, {"eyeglasses": [(600, 300, 700, 400)]}, {}, {0: (True, 0.9, None, None, "")})
    monkeypatch.setitem(server._state, "client", c)
    monkeypatch.setitem(server._state, "pipeline", p)
    h = {"X-LastSeen-Passcode": "pw"}
    tc = TestClient(server.app)
    r = tc.post("/api/guide/start", json={"item": "glasses"}, headers=h).json()
    assert r["action"] == "look" and r["planner"] == "wonder"
    jpg = cv2.imencode(".jpg", scene())[1].tobytes()
    r2 = tc.post("/api/guide/frame", data={"session": r["session"]}, files={"frames": ("g.jpg", jpg, "image/jpeg")}, headers=h).json()
    assert r2["action"] == "done" and r2["found"] and "Found your glasses" in r2["say"]
    assert tc.post("/api/guide/frame", data={"session": r["session"]}, files={"frames": ("g.jpg", jpg, "image/jpeg")}, headers=h).status_code == 409
    r3 = tc.post("/api/ask", json={"text": "guide me to my glasses"}, headers=h).json()
    assert r3["intent"] == "guide" and r3["item"] == "glasses"
