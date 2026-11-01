"""Leaving-home agent: "I'm leaving, check the house."

Muse Wonder plans and runs the walk-through with two tools:

    check_item(item, instruction)   tell the user where to point the phone; the app then
                                    runs that item's VERIFIED check (both witnesses) and
                                    returns the result to Wonder
    finish()                        end the walk-through

Wonder decides the order, re-checks an item when a check abstained (using the
check's own camera guidance), asks the user to turn something off and re-check
when a check found it ON, and gives up on an item after MAX_ATTEMPTS.

What Wonder can NOT do, by design:
- say anything is off. It never sees pixels; it only reads verified results, and
  the user always hears the verified answer itself first, not Wonder's wording.
- write the final summary. That sentence is built by plain code from the verified
  results, so "All clear" only appears when every item on the list was verified.

If Wonder is unavailable or misbehaves, a fixed plan takes over (same rules), so
the walk-through still works; the app shows which planner was used.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

MODEL = "muse-wonder-1.0"
MAX_ATTEMPTS = 3          # per item
MAX_STEPS = 10            # Wonder turns per walk-through
MAX_INSTRUCTION = 140     # characters

SYSTEM = """You run a leaving-home check for a blind or low-vision person, using their phone camera.
You cannot see anything yourself. Each check_item call makes the app run a verified check of one item
(two independent vision models must agree) and returns the result to you.

Rules:
- Call exactly one tool per turn. Check items one at a time, in a sensible order.
- `instruction` is spoken to the user before the check: one short sentence (under 20 words) telling them
  where to point the phone. Never state or guess whether anything is on or off.
- If a result is not verified and has guidance (e.g. "Move closer"), check the same item again and put
  that guidance in the instruction.
- If a result found something ON, ask the user to turn it off, then check the same item again.
- Stop re-checking an item after its attempts run out; move on.
- When every item has been checked or has no attempts left, call finish."""

TOOLS = [
    {"type": "function", "function": {
        "name": "check_item",
        "description": "Ask the user to point the phone at one item, then run its verified check.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["item", "instruction"],
                       "properties": {"item": {"type": "string"},
                                      "instruction": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the leaving-home check.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {}}}},
]

DEFAULT_INSTRUCTION = {
    "stove": "Point the phone at the stove knobs and hold still.",
    "oven display": "Point the phone at the oven display and hold still.",
}


@dataclass
class Step:
    action: str                   # "check" (capture frames for `item`) or "done"
    say: str
    item: str | None = None
    summary: str | None = None
    all_clear: bool = False


@dataclass
class Session:
    id: str
    checklist: list[str]                       # everything the user wants checked
    checkable: list[str]                       # the ones Hold Point can verify
    messages: list[dict] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)   # latest verified result per item
    trace: list[dict] = field(default_factory=list)          # for the model-calls panel
    pending: str | None = None                 # item waiting for frames
    open_call: str | None = None               # Wonder's tool call waiting for that check's result
    planner: str = "wonder"
    wonder_turns: int = 0
    started: float = field(default_factory=time.time)


# ---------------------------------------------------------------- the deterministic parts

def item_status(result: dict | None) -> str:
    if result is None:
        return "unchecked"
    if result.get("all_clear"):
        return "off"
    if result.get("danger"):
        return "on"
    return "unconfirmed"


def summary(s: Session, now: datetime | None = None) -> tuple[str, bool]:
    """The final sentence. Built from verified results only; never from Wonder's words."""
    now = now or datetime.now()
    stamp = now.strftime("%-I:%M %p")
    on = [i for i in s.checkable if item_status(s.results.get(i)) == "on"]
    unconfirmed = [i for i in s.checkable if item_status(s.results.get(i)) in ("unconfirmed", "unchecked")]
    off = [i for i in s.checkable if item_status(s.results.get(i)) == "off"]
    manual = [i for i in s.checklist if i not in s.checkable]

    if on:
        return f"Not all clear. The {' and the '.join(on)} may still be on. Checked {stamp}.", False
    parts = []
    if off:
        parts.append(f"{_join(off).capitalize()} verified off.")
    if unconfirmed:
        parts.append(f"Couldn't confirm the {_join(unconfirmed)}. Please check {'it' if len(unconfirmed) == 1 else 'them'} yourself.")
    if manual:
        parts.append(f"I can't check the {_join(manual)}. Please check {'it' if len(manual) == 1 else 'them'} yourself.")
    if off and not unconfirmed and not manual:
        return f"All clear. {_join(off).capitalize()} verified off. Checked {stamp}.", True
    return " ".join(parts) + f" Checked {stamp}.", False


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fixed_plan(s: Session) -> tuple[str, str | None, str]:
    """Same rules as the prompt, as code. Returns (tool, item, instruction)."""
    for item in s.checkable:
        n = s.attempts.get(item, 0)
        status = item_status(s.results.get(item))
        if status == "off" or n >= MAX_ATTEMPTS:
            continue
        if status == "unchecked":
            return "check_item", item, DEFAULT_INSTRUCTION.get(item, f"Point the phone at the {item}.")
        guidance = s.results[item].get("guidance") or ""
        if status == "on":
            return "check_item", item, f"Please turn the {item} off, then point the phone at it again."
        return "check_item", item, (guidance or f"Point the phone at the {item} again.")
    return "finish", None, ""


# ---------------------------------------------------------------- Wonder

def _result_for_wonder(item: str, result: dict, attempts: int) -> dict:
    return {"item": item, "verified_off": bool(result.get("all_clear")),
            "something_on": bool(result.get("danger")), "verified_answer": result.get("answer"),
            "guidance": result.get("guidance"), "attempts_used": attempts,
            "attempts_left": max(0, MAX_ATTEMPTS - attempts)}


def _ask_wonder(client, s: Session) -> tuple[str, str | None, str]:
    """One Wonder turn. Returns (tool, item, instruction); raises on anything unusable."""
    if s.wonder_turns >= MAX_STEPS:
        raise RuntimeError("step limit reached")
    s.wonder_turns += 1
    client.ledger.check(MODEL)
    t = time.perf_counter()
    resp = client.client.chat.completions.create(
        model=MODEL, messages=s.messages, tools=TOOLS, tool_choice="auto",
        reasoning_effort="low", max_tokens=600)
    client.ledger.record(MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens, tag="agent")
    msg = resp.choices[0].message
    calls = list(msg.tool_calls or [])
    s.messages.append({"role": "assistant", "content": msg.content or "",
                       "tool_calls": [{"id": c.id, "type": "function",
                                       "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                      for c in calls]} if calls else {"role": "assistant", "content": msg.content or ""})
    if not calls:
        raise RuntimeError(f"no tool call (said: {(msg.content or '')[:80]!r})")
    first, rest = calls[0], calls[1:]
    for extra in rest:   # one step at a time: answer any parallel calls so the history stays valid
        s.messages.append({"role": "tool", "tool_call_id": extra.id,
                           "content": json.dumps({"skipped": "one tool call per turn"})})
    name = first.function.name
    try:
        args = json.loads(first.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    s.trace.append({"model": MODEL, "seconds": round(time.perf_counter() - t, 2),
                    "call": f"{name}({args.get('item', '')})" if name == "check_item" else f"{name}()",
                    "said": (args.get("instruction") or "")[:MAX_INSTRUCTION]})
    if name == "finish":
        s.messages.append({"role": "tool", "tool_call_id": first.id, "content": json.dumps({"ok": True})})
        return "finish", None, ""
    item = args.get("item")
    if name != "check_item" or item not in s.checkable:
        raise RuntimeError(f"unusable call {name}({args})")
    if s.attempts.get(item, 0) >= MAX_ATTEMPTS:
        raise RuntimeError(f"asked to re-check {item} with no attempts left")
    s.open_call = first.id  # the tool result is sent after the check runs
    instruction = (args.get("instruction") or "").strip()[:MAX_INSTRUCTION]
    return "check_item", item, instruction or DEFAULT_INSTRUCTION.get(item, f"Point the phone at the {item}.")


def _next(client, s: Session) -> Step:
    tool = item = instruction = None
    if s.planner == "wonder":
        try:
            tool, item, instruction = _ask_wonder(client, s)
        except Exception as e:  # never let a planning problem break the walk-through
            if type(e).__name__ == "BudgetExceeded":
                raise
            s.planner = "fixed"
            s.trace.append({"model": "fixed plan", "seconds": 0, "call": "takes over",
                            "said": f"Wonder unusable: {type(e).__name__}: {e}"[:160]})
    if s.planner == "fixed":
        tool, item, instruction = fixed_plan(s)
        s.trace.append({"model": "fixed plan", "seconds": 0,
                        "call": f"{tool}({item or ''})", "said": instruction})
    if tool == "finish":
        text, clear = summary(s)
        return Step(action="done", say=text, summary=text, all_clear=clear)
    s.pending = item
    return Step(action="check", item=item, say=instruction)


def start(client, checklist: list[str], checkable: list[str], said: str = "") -> tuple[Session, Step]:
    s = Session(id=uuid.uuid4().hex[:12], checklist=checklist,
                checkable=[i for i in checklist if i in checkable])
    s.messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({
            "user_said": said or "I'm leaving, check the house.",
            "items_you_can_check": s.checkable,
            "items_the_user_must_check_themselves": [i for i in checklist if i not in s.checkable],
            "max_attempts_per_item": MAX_ATTEMPTS})},
    ]
    if not s.checkable:
        text, clear = summary(s)
        return s, Step(action="done", say=text, summary=text, all_clear=clear)
    return s, _next(client, s)


def submit(client, s: Session, result: dict) -> Step:
    """Feed the verified result for the pending item back and get the next step."""
    item = s.pending
    if item is None:
        raise ValueError("No item is waiting for a check.")
    s.pending = None
    s.attempts[item] = s.attempts.get(item, 0) + 1
    s.results[item] = result
    call_id = s.open_call
    if s.planner == "wonder" and call_id:
        s.messages.append({"role": "tool", "tool_call_id": call_id,
                           "content": json.dumps(_result_for_wonder(item, result, s.attempts[item]))})
        s.open_call = None
    step = _next(client, s)
    # The user hears the verified answer first; Wonder's wording only says where to point next.
    step.say = (result.get("answer", "") + " " + step.say).strip() if step.action == "check" else step.say
    return step
