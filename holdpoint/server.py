"""Hold Point live app: serves the phone page and runs checks.

    export HOLDPOINT_PASSCODE=...   # required: the page asks for it once
    uvicorn holdpoint.server:app --port 8000

The phone reaches this through an HTTPS tunnel (phone browsers only allow the
camera on HTTPS pages). Because a tunnel URL is public, every check needs the
passcode, only one check runs at a time, and the budget cap in budget.py applies.
Each answer is appended to home/check_log.jsonl (the timestamped check record).
Each check's frames and readings are saved to home/live/<time>/ (stays on this
laptop, git-ignored) so real checks become evaluation data. Set
HOLDPOINT_SAVE_FRAMES=0 to turn that off.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .budget import BudgetExceeded
from .check import run_stove_check
from .oven import run_oven_check
from .decision import DecisionConfig
from .profile import HOME_DIR, load_stove
from .setup_live import SetupFailed, setup_from_frames
from .voice import HELP, VoiceError, parse_intent, transcribe
from . import agent as agent_mod
from pydantic import BaseModel

STATIC = Path(__file__).resolve().parent / "static"
MAX_FRAMES = 6
CFG = DecisionConfig(min_visible_frames=3)

app = FastAPI(title="Hold Point")

# Last Seen ("where did I leave it?") runs inside the same server: page at /lastseen/,
# same passcode, same spend cap, data in home/lastseen/.
from lastseen import server as lastseen_server   # noqa: E402
from fastapi.responses import RedirectResponse   # noqa: E402
app.mount("/lastseen", lastseen_server.app)


@app.get("/lastseen", include_in_schema=False)
def _lastseen_slash():
    return RedirectResponse("/lastseen/")
_busy = asyncio.Lock()
_state: dict = {}


def _client():
    if "client" not in _state:
        from .meta_client import MetaClient
        _state["client"] = MetaClient()
    return _state["client"]


def _stove():
    try:
        return load_stove()
    except SystemExit as e:  # load_stove exits with a helpful message for the command line
        raise HTTPException(503, str(e))


def _check_passcode(given: str | None) -> None:
    expected = os.environ.get("HOLDPOINT_PASSCODE")
    if not expected:
        raise HTTPException(503, "Server has no passcode set. Set HOLDPOINT_PASSCODE and restart.")
    if given != expected:
        raise HTTPException(401, "Wrong passcode.")


def log_check(result: dict, home: Path = HOME_DIR) -> str:
    """Save a one-line, timestamped record of the check. Returns the record's sentence."""
    now = datetime.now()
    sentence = f"Checked {now.strftime('%-I:%M %p')}: {result['answer']}"
    home.mkdir(parents=True, exist_ok=True)
    with (home / "check_log.jsonl").open("a") as f:
        f.write(json.dumps({"time": now.isoformat(timespec="seconds"), "answer": result["answer"],
                            "all_clear": result["all_clear"], "cost_usd": result["cost_usd"]}) + "\n")
    return sentence


def save_frames(images: list[np.ndarray], result: dict, home: Path = HOME_DIR) -> Path | None:
    """Keep the frames and readings of a live check for the evaluation."""
    if os.environ.get("HOLDPOINT_SAVE_FRAMES", "1") == "0":
        return None
    folder = home / "live" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(str(folder / f"frame{i + 1}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (folder / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k != "overlay"}, indent=2))
    return folder


async def _read_images(frames: list[UploadFile]) -> list[np.ndarray]:
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise HTTPException(400, f"Send 1 to {MAX_FRAMES} frames.")
    images = []
    for f in frames:
        img = cv2.imdecode(np.frombuffer(await f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "A frame couldn't be read as an image.")
        images.append(img)
    return images


@app.get("/")
def page():
    """Glasses-style, hands-free page: passcode + Start once, then voice only."""
    return FileResponse(STATIC / "glasses.html")


@app.get("/manual")
def manual():
    """The page with buttons: stove setup, evaluation mode, and a fallback."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status(x_holdpoint_passcode: str | None = Header(None)):
    _check_passcode(x_holdpoint_passcode)
    stove, _ = _stove()
    ledger = _client().ledger
    return {"knobs": [k["name"] for k in stove["knobs"]], "spent_usd": round(ledger.spent(), 4),
            "cap_usd": ledger.cap}


TRUTH = re.compile(r"^(all_off|(on|slight):knob_[1-6]|oven_off|oven_on)$")
ITEMS = {"stove", "oven"}


@app.post("/api/check")
async def check(frames: list[UploadFile] = File(...), truth: str | None = None, item: str = "stove",
                x_holdpoint_passcode: str | None = Header(None)):
    """truth (evaluation only): what the stove REALLY is, e.g. "all_off", "on:knob_3",
    "slight:knob_1". Saved with the check's frames; never used to decide the answer."""
    _check_passcode(x_holdpoint_passcode)
    if truth is not None and not TRUTH.match(truth):
        raise HTTPException(400, "truth must be all_off, on:knob_N, slight:knob_N, oven_off or oven_on")
    if item not in ITEMS:
        raise HTTPException(400, "item must be stove or oven")
    if _busy.locked():
        raise HTTPException(429, "A check is already running.")
    async with _busy:
        images = await _read_images(frames)
        cfg = CFG if len(images) >= CFG.min_visible_frames else DecisionConfig(min_visible_frames=len(images))
        t = time.perf_counter()
        try:
            if item == "oven":
                result = await run_in_threadpool(run_oven_check, _client(), images, cfg)
            else:
                stove, refs = _stove()
                result = await run_in_threadpool(run_stove_check, _client(), images, stove, refs, cfg)
        except BudgetExceeded as e:
            raise HTTPException(402, f"Budget cap reached: {e}")
        result["record"] = log_check(result)
        result["server_seconds"] = round(time.perf_counter() - t, 2)
        save_frames(images, {**result, "truth": truth} if truth else result)
        return result


# What the phone does after each spoken command. "check" = capture frames and run a stove check.
INTENT_REPLY = {
    "check_stove": ("check", "Checking the stove. Hold still."),
    "leaving": ("agent", "Okay. Let's check the house."),
    "check_oven": (None, "I can't check the oven display yet. Please check it yourself."),
    "stop": ("stop", "Okay, stopped."),
    None: (None, HELP),
}


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...), x_holdpoint_passcode: str | None = Header(None)):
    """Spoken command (WAV) -> Muse Voice Transcribe -> what to do next."""
    _check_passcode(x_holdpoint_passcode)
    data = await audio.read()
    t = time.perf_counter()
    try:
        heard = await run_in_threadpool(transcribe, _client(), data)
    except VoiceError as e:
        raise HTTPException(422, str(e))
    except BudgetExceeded as e:
        raise HTTPException(402, f"Budget cap reached: {e}")
    ls = lastseen_question(heard["transcript"]) if heard["transcript"] else None
    if ls is not None:                         # "where are my keys?" / "guide me to my keys" (Last Seen)
        intent, item = ls
        out = {"transcript": heard["transcript"], "intent": intent, "item": item,
               "audio_seconds": heard["seconds"], "seconds": round(time.perf_counter() - t, 2)}
        if intent == "where":
            return {**out, "action": "where", **lastseen_where(item)}
        return {**out, "action": "guide", "reply": "Okay. Let's find them." if item.endswith("s") else "Okay. Let's find it."}
    intent = parse_intent(heard["transcript"]) if heard["transcript"] else None
    action, reply = INTENT_REPLY[intent]
    if not heard["transcript"]:
        reply = "I didn't hear anything. Tap, speak, then tap again."
    return {"transcript": heard["transcript"], "intent": intent, "action": action, "reply": reply,
            "audio_seconds": heard["seconds"], "seconds": round(time.perf_counter() - t, 2)}


@app.post("/api/setup")
async def setup(frames: list[UploadFile] = File(...), knobs: int = 4,
                x_holdpoint_passcode: str | None = Header(None)):
    """Save every knob's OFF direction from the phone. All knobs must be OFF."""
    _check_passcode(x_holdpoint_passcode)
    if not 1 <= knobs <= 6:
        raise HTTPException(400, "knobs must be 1 to 6.")
    if _busy.locked():
        raise HTTPException(429, "A check is already running.")
    async with _busy:
        images = await _read_images(frames)
        try:
            return await run_in_threadpool(setup_from_frames, _client(), images, knobs)
        except SetupFailed as e:
            raise HTTPException(422, str(e))
        except BudgetExceeded as e:
            raise HTTPException(402, f"Budget cap reached: {e}")


# ---------------------------------------------------------------- leaving-home agent

# The oven display reader (holdpoint/oven.py) is experimental: on 4 Nov its LED-shape witness
# could not read 16-pixel digits, so it is not part of the walk-through yet.
CHECKABLE = {"stove"}                    # items with a verified check today
DEFAULT_CHECKLIST = ["stove"]


def leaving_reminder(find: dict | None = None) -> list[str]:
    """Names of the user's Last Seen items, for "Don't forget your keys, glasses and wallet."
    (items already being found in the walk-through are left out)."""
    from lastseen.home import Home as LSHome
    return [i.name for i in LSHome.load().items if i.id not in (find or {})]


def lastseen_question(text: str) -> tuple[str, str | None] | None:
    """ "where are my keys?" / "guide me to my wallet" -> ("where" | "guide", item id).
    Checked BEFORE Hold Point's own commands, so "where did I leave my keys" is not "I'm leaving"."""
    from lastseen.home import Home as LSHome
    from lastseen.voice import parse_intent as ls_intent
    home = LSHome.load()
    if not home.items:
        return None
    intent, item = ls_intent(text, home.item_names())
    if intent in ("where", "guide") and item is not None:
        return intent, item
    return None


def lastseen_where(item: str) -> dict:
    from lastseen.answer import answer_where
    from lastseen.home import Home as LSHome
    from lastseen.memory import MemoryStore
    home = LSHome.load()
    a = answer_where(item, MemoryStore().for_item(item), home)
    text = a["text"]
    if a["status"] != "never_seen":
        text += f" Say: guide me to my {home.item(item).name}."
    return {"reply": text, "status": a["status"]}


def leaving_find(environ=os.environ) -> dict[str, str]:
    """Belongings to find at the end of the leaving-home check (optional).

    Switched on by HOLDPOINT_LEAVING_FIND, e.g. "keys" or "keys,glasses" (Last Seen
    item ids). Only items enrolled in Last Seen count. Returns {item: where it was
    last seen, as a spoken sentence}. Unset or empty -> {} and the check is unchanged."""
    wanted = [w.strip().lower() for w in environ.get("HOLDPOINT_LEAVING_FIND", "").split(",") if w.strip()]
    if not wanted:
        return {}
    from lastseen.find_check import last_known
    from lastseen.home import Home as LSHome
    from lastseen.memory import MemoryStore
    home, store = LSHome.load(), MemoryStore()
    enrolled = {i.id for i in home.items}
    return {w: last_known(w, home, store)[1] for w in wanted if w in enrolled}
_sessions: dict[str, agent_mod.Session] = {}


def leaving_checklist(home: Path = HOME_DIR) -> list[str]:
    """The user's own list, from home/profile.json ("leaving_checklist"), else the default."""
    try:
        items = json.loads((home / "profile.json").read_text()).get("leaving_checklist")
    except (OSError, ValueError):
        items = None
    return [str(i).lower() for i in items] if items else list(DEFAULT_CHECKLIST)


class AgentStart(BaseModel):
    said: str = ""


def _agent_reply(s: agent_mod.Session, step: agent_mod.Step, check: dict | None = None) -> dict:
    out = {"session": s.id, "action": step.action, "item": step.item, "say": step.say,
           "kind": "find" if step.item in s.find_where else "check",
           "planner": s.planner, "trace": s.trace, "check": check}
    if step.action == "done":
        _sessions.pop(s.id, None)
        log_check({"answer": step.summary, "all_clear": step.all_clear, "cost_usd": 0.0})
        out.update(summary=step.summary, all_clear=step.all_clear, record=step.summary)
    return out


@app.post("/api/agent/start")
async def agent_start(body: AgentStart, x_holdpoint_passcode: str | None = Header(None)):
    _check_passcode(x_holdpoint_passcode)
    find = leaving_find()
    try:
        s, step = await run_in_threadpool(agent_mod.start, _client(), leaving_checklist(), CHECKABLE, body.said,
                                          {"stove": "kitchen"}, find, leaving_reminder(find))
    except BudgetExceeded as e:
        raise HTTPException(402, f"Budget cap reached: {e}")
    _sessions.clear()                    # one walk-through at a time
    _sessions[s.id] = s
    return _agent_reply(s, step)


@app.post("/api/agent/step")
async def agent_step(session: str, frames: list[UploadFile] = File(...),
                     x_holdpoint_passcode: str | None = Header(None)):
    _check_passcode(x_holdpoint_passcode)
    s = _sessions.get(session)
    if s is None or s.pending is None:
        raise HTTPException(409, "That walk-through has ended. Say: I'm leaving, check the house.")
    if _busy.locked():
        raise HTTPException(429, "A check is already running.")
    async with _busy:
        images = await _read_images(frames)
        try:
            if s.pending == "stove":
                stove, refs = _stove()
                result = await run_in_threadpool(run_stove_check, _client(), images, stove, refs, CFG)
            elif s.pending == "oven display":
                result = await run_in_threadpool(run_oven_check, _client(), images, CFG)
            elif s.pending in s.find_where:  # a belonging: Last Seen's verified find
                from lastseen.find_check import run_find_check
                result = await run_in_threadpool(run_find_check, _client(), images, s.pending)
            else:                        # cannot happen: only CHECKABLE items are ever pending
                raise HTTPException(500, f"No check for {s.pending}.")
            if result.get("kind") != "find":   # home/live/ is Hold Point's evaluation data: appliance checks only
                save_frames(images, result)
            step = await run_in_threadpool(agent_mod.submit, _client(), s, result)
        except BudgetExceeded as e:
            raise HTTPException(402, f"Budget cap reached: {e}")
    return _agent_reply(s, step, check=result)


# ---------------------------------------------------------------- glasses mode: is the stove in view yet?

@app.post("/api/scout")
async def scout(frame: UploadFile = File(...), x_holdpoint_passcode: str | None = Header(None)):
    """One frame -> SAM 3.1 only (~$0.0025). Says whether all the knobs are in view and big
    enough for a full verified check, and if not, which way to move. Never says on/off."""
    _check_passcode(x_holdpoint_passcode)
    if _busy.locked():
        raise HTTPException(429, "A check is already running.")
    async with _busy:
        img = (await _read_images([frame]))[0]
        stove, _ = _stove()
        try:
            return await run_in_threadpool(scout_frame, _client(), img, stove)
        except BudgetExceeded as e:
            raise HTTPException(402, f"Budget cap reached: {e}")


def scout_frame(client, img: np.ndarray, stove: dict, segment=None) -> dict:
    from .meta_client import prepare_frame
    from .witness_sam import MIN_RADIUS_PX, assign_knobs, knob_circle, pick_row, segment_image
    segment = segment or segment_image
    frame = prepare_frame(img)
    n = len(stove["knobs"])
    t = time.perf_counter()
    found = pick_row(segment(client, frame, stove["sam_prompt"], tag="scout"), n)
    assigned = assign_knobs(found, [k["id"] for k in stove["knobs"]], frame.shape[1])
    radii = [knob_circle(o)[2] for o in found]
    r = float(np.median(radii)) if radii else 0.0
    out = {"knobs_seen": len(found), "expected": n, "radius_px": round(r, 1),
           "seconds": round(time.perf_counter() - t, 2), "ready": False}
    if not found:
        out["say"] = "I can't see the stove yet. Keep looking toward the stove."
    elif len(assigned) < n or len(found) < n:
        xs = [o.centre[0] for o in found]
        w = frame.shape[1]
        side = "left" if min(xs) < 0.15 * w and max(xs) < 0.85 * w else "right" if max(xs) > 0.85 * w else None
        out["say"] = f"I can see part of the stove. Turn a little {side}." if side else "Step back a little so I can see all the knobs."
    elif r < MIN_RADIUS_PX * 1.5:
        out["say"] = "I can see the stove. Move closer."
    else:
        out["ready"], out["say"] = True, "I can see all the knobs. Checking. Hold still."
    return out
