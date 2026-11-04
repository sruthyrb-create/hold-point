"""Guide me to it: Muse Wonder walks the user to where the item was last seen.

Wonder plans the search with two tools:

    look_at(place, instruction)   say where to go and what to face; the app then looks
                                  through the camera (up to LOOK_VIEWS views, both
                                  witnesses) and returns what was verified
    finish()                      end the search

Wonder only sees the place list built from memory (newest first, with how sure
each one is) and verified look results. It decides the order and words each
instruction. It can NOT say the item is found: "Found your glasses ..." comes
only from a verified look, and the final sentence is built by plain code.
If Wonder is unavailable or misbehaves, a fixed plan with the same rules takes over.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from .answer import age_phrase
from .home import Home
from .memory import CONFIRMED, Sighting
from .spatial import place_phrase

MODEL = "muse-wonder-1.0"
MAX_ATTEMPTS = 2          # looks per place
MAX_STEPS = 8             # Wonder turns per search
LOOK_VIEWS = 3            # camera views per look before reporting back
MAX_INSTRUCTION = 140
HERE = "here"             # always-available last resort: look around where you are
SEEN = "last_seen_spot"   # newest sighting had no known landmark; described in Spark's words

SYSTEM = """You guide a blind or low-vision person to one of their belongings, using the camera on their phone or glasses.
You cannot see anything yourself. Each look_at call speaks your instruction, then the app looks through the camera
(two independent vision models must agree) and returns what it verified.

Rules:
- Call exactly one tool per turn.
- Start with the most likely place: usually the newest CONFIRMED place. Use the others if that fails.
- `instruction` is spoken: one short sentence under 20 words. Name the room and the landmark to face, e.g.
  "Go to the kitchen and face the stove, then turn slowly." Never say or guess that the item is there.
- If a look saw the landmark but not the item, say to lower the phone or turn slowly and look once more,
  or move to the next place.
- If a look did not see the landmark, the user may not be there yet: repeat with clearer directions.
- When the item is found, or every place has no attempts left, call finish."""

TOOLS = [
    {"type": "function", "function": {
        "name": "look_at", "description": "Speak an instruction, then look for the item through the camera.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["place", "instruction"],
                       "properties": {"place": {"type": "string"}, "instruction": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "End the search.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {}}}},
]


@dataclass
class Place:
    id: str                      # landmark id, or HERE
    room: str | None
    landmark: str | None         # landmark name
    phrase: str                  # "left of the stove, in the kitchen"
    age_s: float | None
    confirmed: bool


@dataclass
class Step:
    action: str                  # "look" (send frames) or "done"
    say: str
    place: str | None = None
    found: bool = False


@dataclass
class Session:
    id: str
    item: str
    item_name: str
    places: list[Place]
    messages: list[dict] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    last_look: dict[str, dict] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    pending: str | None = None
    open_call: str | None = None
    views: int = 0               # camera views so far in the current look
    landmark_seen: bool = False  # in any view of the current look
    found: dict | None = None
    planner: str = "wonder"
    wonder_turns: int = 0


# ---------------------------------------------------------------- deterministic parts

def build_places(item_id: str, sightings: list[Sighting], home: Home, now: float | None = None) -> list[Place]:
    """Where to look, newest first; one entry per landmark; always ends with HERE."""
    now = now if now is not None else time.time()
    out, seen = [], set()
    mine = sorted([s for s in sightings if s.item == item_id], key=lambda s: -s.t)
    newest = mine[0] if mine else None
    if newest is not None and (newest.landmark is None or home.landmark(newest.landmark) is None) and newest.seen_on:
        # newest sighting had no known landmark: start where it was, described by Spark (unconfirmed)
        out.append(Place(SEEN, None, newest.seen_on, f"on what looked like a {newest.seen_on}",
                         now - newest.t, False))
    for s in mine:
        lm = home.landmark(s.landmark) if s.landmark else None
        if lm is None or lm.id in seen:
            continue
        seen.add(lm.id)
        rel = s.relation or "near"
        out.append(Place(lm.id, lm.room, lm.name, place_phrase(rel, lm.name, lm.room), now - s.t,
                         s.status == CONFIRMED))
    # newest first: things move, so the latest sighting is the best bet (Wonder sees "confirmed" too)
    out.append(Place(HERE, None, None, "around you", None, False))
    return out


def default_instruction(p: Place) -> str:
    if p.id == HERE:
        return "Hold the phone at chest height and turn slowly where you are."
    if p.id == SEEN:
        return f"Go to the {p.landmark}, if you know where that is, then turn slowly."
    return f"Go to the {p.room} and face the {p.landmark}, then turn slowly."


def fixed_plan(s: Session) -> tuple[str, str | None, str]:
    if s.found:
        return "finish", None, ""
    for p in s.places:
        n = s.attempts.get(p.id, 0)
        if n >= MAX_ATTEMPTS:
            continue
        last = s.last_look.get(p.id)
        if last is None:
            return "look_at", p.id, default_instruction(p)
        if last["landmark_seen"] and p.id not in (HERE, SEEN):
            return "look_at", p.id, f"Lower the phone a little and turn slowly near the {p.landmark}."
        return "look_at", p.id, default_instruction(p)
    return "finish", None, ""


def final_sentence(s: Session) -> tuple[str, bool]:
    if s.found:
        return s.found["text"], True
    tried = [p.phrase for p in s.places if p.id not in (HERE, SEEN) and s.attempts.get(p.id)]
    where = f" at {', or '.join(tried)}" if tried else ""
    return (f"I couldn't find your {s.item_name}{where}. They may have moved. "
            "Turn on watching and I'll remember where I next see them."
            if s.item_name.endswith("s") else
            f"I couldn't find your {s.item_name}{where}. It may have moved. "
            "Turn on watching and I'll remember where I next see it."), False


def _place_for_wonder(p: Place) -> dict:
    where = {HERE: "where the user is standing now",
             SEEN: f"unconfirmed: it looked like a {p.landmark}; no known landmark was in view"}.get(p.id, p.phrase)
    return {"place": p.id, "where": where,
            "last_seen": age_phrase(p.age_s) if p.age_s is not None else None,
            "confirmed": p.confirmed}


# ---------------------------------------------------------------- Wonder

def _ask_wonder(client, s: Session) -> tuple[str, str | None, str]:
    if s.wonder_turns >= MAX_STEPS:
        raise RuntimeError("step limit reached")
    s.wonder_turns += 1
    client.ledger.check(MODEL)
    t = time.perf_counter()
    resp = client.client.chat.completions.create(
        model=MODEL, messages=s.messages, tools=TOOLS, tool_choice="auto",
        reasoning_effort="low", max_tokens=500)
    client.ledger.record(MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens, tag="guide")
    msg = resp.choices[0].message
    calls = list(msg.tool_calls or [])
    s.messages.append({"role": "assistant", "content": msg.content or "",
                       "tool_calls": [{"id": c.id, "type": "function",
                                       "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                      for c in calls]} if calls else {"role": "assistant", "content": msg.content or ""})
    if not calls:
        raise RuntimeError(f"no tool call (said: {(msg.content or '')[:80]!r})")
    first = calls[0]
    for extra in calls[1:]:
        s.messages.append({"role": "tool", "tool_call_id": extra.id,
                           "content": json.dumps({"skipped": "one tool call per turn"})})
    name = first.function.name
    try:
        args = json.loads(first.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    s.trace.append({"model": MODEL, "seconds": round(time.perf_counter() - t, 2),
                    "call": f"{name}({args.get('place', '')})",
                    "said": (args.get("instruction") or "")[:MAX_INSTRUCTION]})
    if name == "finish":
        s.messages.append({"role": "tool", "tool_call_id": first.id, "content": json.dumps({"ok": True})})
        return "finish", None, ""
    place = args.get("place")
    ids = {p.id for p in s.places}
    if name != "look_at" or place not in ids:
        raise RuntimeError(f"unusable call {name}({args})")
    if s.attempts.get(place, 0) >= MAX_ATTEMPTS:
        raise RuntimeError(f"asked to look at {place} with no attempts left")
    s.open_call = first.id
    instr = (args.get("instruction") or "").strip()[:MAX_INSTRUCTION]
    p = next(p for p in s.places if p.id == place)
    return "look_at", place, instr or default_instruction(p)


def _next(client, s: Session) -> Step:
    tool = place = instr = None
    if s.found:
        tool = "finish"           # found: no need to ask anyone
    elif s.planner == "wonder":
        try:
            tool, place, instr = _ask_wonder(client, s)
        except Exception as e:
            if type(e).__name__ == "BudgetExceeded":
                raise
            s.planner = "fixed"
            s.trace.append({"model": "fixed plan", "seconds": 0, "call": "takes over",
                            "said": f"Wonder unusable: {type(e).__name__}: {e}"[:160]})
    if tool is None:
        tool, place, instr = fixed_plan(s)
        s.trace.append({"model": "fixed plan", "seconds": 0, "call": f"{tool}({place or ''})", "said": instr})
    if tool == "finish":
        text, found = final_sentence(s)
        return Step("done", text, found=found)
    s.pending, s.views, s.landmark_seen = place, 0, False
    return Step("look", instr, place=place)


def start(client, item_id: str, home: Home, sightings: list[Sighting], said: str = "",
          now: float | None = None) -> tuple[Session, Step]:
    it = home.item(item_id)
    places = build_places(item_id, sightings, home, now)
    s = Session(uuid.uuid4().hex[:12], item_id, it.name, places)
    s.messages = [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps({
                      "user_said": said or f"Help me find my {it.name}.",
                      "item": it.name, "places_newest_first": [_place_for_wonder(p) for p in places],
                      "max_attempts_per_place": MAX_ATTEMPTS})}]
    return s, _next(client, s)


def add_view(client, s: Session, look: dict) -> Step | None:
    """One camera view during a look. `look` = Pipeline.find(...) output.
    Returns None to keep looking (send another view), or the next Step."""
    if s.pending is None:
        raise ValueError("No look is running.")
    s.views += 1
    s.landmark_seen = s.landmark_seen or bool(look.get("landmark_seen"))
    if look.get("found"):
        s.found = look
    elif s.views < LOOK_VIEWS:
        return None
    place = s.pending
    s.pending = None
    s.attempts[place] = s.attempts.get(place, 0) + 1
    s.last_look[place] = {"found": bool(s.found), "landmark_seen": s.landmark_seen, "views": s.views}
    if s.planner == "wonder" and s.open_call:
        s.messages.append({"role": "tool", "tool_call_id": s.open_call, "content": json.dumps({
            "place": place, "item_found": bool(s.found), "landmark_in_view": s.landmark_seen,
            "camera_views": s.views, "attempts_left": max(0, MAX_ATTEMPTS - s.attempts[place])})})
        s.open_call = None
    return _next(client, s)
