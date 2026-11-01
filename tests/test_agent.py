"""Leaving-home agent with a scripted fake Muse Wonder (no key, no cost)."""
import json
from datetime import datetime
from types import SimpleNamespace

from holdpoint import agent
from holdpoint.budget import Ledger, Price
from holdpoint.meta_client import MetaClient

OFF = {"answer": "Yes, all four off.", "all_clear": True, "danger": False, "guidance": None}
UNSURE = {"answer": "Three off. Can't confirm knob 1. Move closer.", "all_clear": False, "danger": False,
          "guidance": "Move closer."}
ON = {"answer": "No. knob 3 is ON. Others unconfirmed.", "all_clear": False, "danger": True, "guidance": None}


def call(name, **args):
    return SimpleNamespace(id=f"c{call.n}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
call.n = 0


class FakeWonder:
    """Replies with the next scripted tool call; records what it was sent."""
    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.sent.append(json.loads(json.dumps(kw["messages"])))
        nxt = self.script.pop(0)
        calls = nxt if isinstance(nxt, list) else ([nxt] if nxt else [])
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=900, completion_tokens=60),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls))])


def client(tmp_path, script):
    ledger = Ledger(cap_usd=1.0, ledger_file=tmp_path / "l.jsonl", prices={"muse-wonder-1.0": Price(4.0, 20.0)})
    return MetaClient(ledger=ledger, client=FakeWonder(script), cache_dir=tmp_path / "c")


def test_wonder_rechecks_then_finishes(tmp_path):
    c = client(tmp_path, [call("check_item", item="stove", instruction="Point at the stove knobs."),
                          call("check_item", item="stove", instruction="Move closer to the knobs."),
                          call("finish")])
    s, step = agent.start(c, ["stove"], {"stove"}, "I'm leaving")
    assert step.action == "check" and step.item == "stove" and step.say == "Point at the stove knobs."
    step = agent.submit(c, s, UNSURE)
    assert step.action == "check" and step.say.startswith(UNSURE["answer"])   # verified answer first
    tool_msg = c.client.sent[-1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["guidance"] == "Move closer."
    step = agent.submit(c, s, OFF)
    assert step.action == "done" and step.all_clear and step.summary.startswith("All clear. Stove verified off.")
    assert s.planner == "wonder" and c.ledger.spent() > 0


def test_finish_too_early_is_not_all_clear(tmp_path):
    c = client(tmp_path, [call("finish")])           # Wonder quits without checking anything
    s, step = agent.start(c, ["stove"], {"stove"})
    assert step.action == "done" and not step.all_clear and "Couldn't confirm the stove" in step.summary


def test_something_on_is_never_all_clear(tmp_path):
    c = client(tmp_path, [call("check_item", item="stove", instruction="x"), call("finish")])
    s, step = agent.start(c, ["stove"], {"stove"})
    step = agent.submit(c, s, ON)
    assert not step.all_clear and step.summary.startswith("Not all clear. The stove may still be on.")


def test_items_without_a_check_are_named_for_the_user(tmp_path):
    c = client(tmp_path, [call("check_item", item="stove", instruction="x"), call("finish")])
    s, step = agent.start(c, ["stove", "back door"], {"stove"})
    step = agent.submit(c, s, OFF)
    assert not step.all_clear
    assert "Stove verified off." in step.summary and "I can't check the back door" in step.summary


def test_bad_wonder_reply_hands_over_to_fixed_plan(tmp_path):
    c = client(tmp_path, [call("check_item", item="oven", instruction="x")])   # not on the list
    s, step = agent.start(c, ["stove"], {"stove"})
    assert s.planner == "fixed" and step.action == "check" and step.item == "stove"
    step = agent.submit(c, s, UNSURE)
    assert step.action == "check" and "Move closer." in step.say
    step = agent.submit(c, s, OFF)
    assert step.action == "done" and step.all_clear


def test_no_tool_call_hands_over_to_fixed_plan(tmp_path):
    c = client(tmp_path, [None])
    s, step = agent.start(c, ["stove"], {"stove"})
    assert s.planner == "fixed" and step.item == "stove"


def test_attempts_run_out(tmp_path):
    c = client(tmp_path, [call("check_item", item="stove", instruction="x")] * 4)
    s, step = agent.start(c, ["stove"], {"stove"})
    for _ in range(agent.MAX_ATTEMPTS - 1):
        step = agent.submit(c, s, UNSURE)
        assert step.action == "check"
    step = agent.submit(c, s, UNSURE)               # Wonder asks a 4th time: refused, fixed plan finishes
    assert step.action == "done" and not step.all_clear and s.planner == "fixed"


def test_parallel_calls_get_answered(tmp_path):
    c = client(tmp_path, [[call("check_item", item="stove", instruction="a"), call("finish")], call("finish")])
    s, step = agent.start(c, ["stove"], {"stove"})
    ids = [m.get("tool_call_id") for m in s.messages if m["role"] == "tool"]
    assert len(ids) == 1                            # the extra call was answered "skipped"
    step = agent.submit(c, s, OFF)
    assert step.all_clear


def test_summary_wording():
    s = agent.Session(id="x", checklist=["stove", "oven display"], checkable=["stove", "oven display"])
    s.results = {"stove": OFF, "oven display": OFF}
    text, clear = agent.summary(s, now=datetime(2026, 11, 1, 21, 2))
    assert clear and text == "All clear. Stove and oven display verified off. Checked 9:02 PM."


def test_server_agent_flow(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from holdpoint import server

    c = client(tmp_path, [call("check_item", item="stove", instruction="Point at the knobs."), call("finish")])
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "p")
    monkeypatch.setattr(server, "_client", lambda: c)
    monkeypatch.setattr(server, "leaving_checklist", lambda: ["stove"])
    monkeypatch.setattr(server, "_stove", lambda: ({}, {}))
    monkeypatch.setattr(server, "run_stove_check", lambda *a, **k: dict(OFF))
    monkeypatch.setattr(server, "save_frames", lambda *a: None)
    monkeypatch.setattr(server, "log_check", lambda r: "logged")
    http, h = TestClient(server.app), {"X-Holdpoint-Passcode": "p"}
    r = http.post("/api/agent/start", json={"said": "I'm leaving"}, headers=h).json()
    assert r["action"] == "check" and r["item"] == "stove"
    img = cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))[1].tobytes()
    r = http.post(f"/api/agent/step?session={r['session']}", headers=h,
                  files=[("frames", ("f.jpg", img, "image/jpeg"))]).json()
    assert r["action"] == "done" and r["all_clear"] and r["record"].startswith("All clear.")
    assert http.post(f"/api/agent/step?session={r['session']}", headers=h,
                     files=[("frames", ("f.jpg", img, "image/jpeg"))]).status_code == 409
