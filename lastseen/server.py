"""Last Seen web server: the phone page plus a small JSON API.

Run:  uvicorn lastseen.server:app --port 8001
Then: ~/cloudflared tunnel --url http://localhost:8001  (HTTPS for the phone camera)

Every API call needs the X-LastSeen-Passcode header (env LASTSEEN_PASSCODE).
One model job runs at a time, and the budget cap in budget.py applies.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import guide as guide_mod
from . import voice
from .answer import answer_where
from .budget import BudgetExceeded, UnpricedModel
from .enrol import EnrolError, add_landmark, enrol_item
from .home import HOME_DIR, Home
from .memory import MemoryStore
from .watch import Pipeline

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Last Seen")
_busy = asyncio.Lock()
_state: dict = {"pipeline": None, "watching": False}


def _client():
    from .meta_client import MetaClient
    if "client" not in _state:
        _state["client"] = MetaClient()
    return _state["client"]


def _pipeline() -> Pipeline:
    if _state["pipeline"] is None:
        _state["pipeline"] = Pipeline(_client(), Home.load(), MemoryStore())
    return _state["pipeline"]


def _reload_home() -> None:
    if _state["pipeline"] is not None:
        _state["pipeline"].home = Home.load()


def _check_passcode(given: str | None) -> None:
    want = os.environ.get("LASTSEEN_PASSCODE") or os.environ.get("HOLDPOINT_PASSCODE")
    if not want:
        raise HTTPException(503, "Server has no passcode set. Set LASTSEEN_PASSCODE (or HOLDPOINT_PASSCODE) and restart.")
    if given != want:
        raise HTTPException(401, "Wrong passcode.")


async def _images(frames: list[UploadFile], lo: int, hi: int) -> list[np.ndarray]:
    if not lo <= len(frames) <= hi:
        raise HTTPException(400, f"Send {lo} to {hi} frames.")
    out = []
    for f in frames:
        img = cv2.imdecode(np.frombuffer(await f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "A frame couldn't be read as an image.")
        out.append(img)
    return out


async def _run(fn, *args):
    if _busy.locked():
        raise HTTPException(429, "Busy with the previous frame.")
    async with _busy:
        try:
            return await asyncio.to_thread(fn, *args)
        except (BudgetExceeded, UnpricedModel) as e:
            raise HTTPException(402, f"Budget: {e}")
        except EnrolError as e:
            raise HTTPException(422, str(e))


def _spent() -> str:
    return _client().ledger.summary()


@app.get("/", response_class=HTMLResponse)
def page():
    return (STATIC / "index.html").read_text()


@app.get("/api/status")
def status(x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    home = Home.load()
    p = _pipeline()
    return {"items": [{"id": i.id, "name": i.name, "description": i.description} for i in home.items],
            "landmarks": [l.__dict__ for l in home.landmarks], "watching": _state["watching"],
            "frames_seen": p.state.frames_seen, "frames_processed": p.state.frames_processed,
            "spent": _spent()}


@app.post("/api/enrol")
async def enrol(frames: list[UploadFile] = File(...), name: str = Form(...), sam_phrase: str = Form(""),
                description: str = Form(""), aliases: str = Form(""),
                x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    imgs = await _images(frames, 2, 4)
    item = await _run(enrol_item, _client(), Home.load(), imgs, name, sam_phrase, description,
                      [a for a in aliases.split(",") if a.strip()])
    _reload_home()
    return {"item": item.__dict__, "spent": _spent()}


@app.post("/api/landmark")
async def landmark(frames: list[UploadFile] = File(...), name: str = Form(...), room: str = Form(...),
                   sam_phrase: str = Form(""), x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    imgs = await _images(frames, 1, 1)
    lm = await _run(add_landmark, _client(), Home.load(), imgs[0], name, room, sam_phrase)
    _reload_home()
    return {"landmark": lm.__dict__, "spent": _spent()}


class Watching(BaseModel):
    on: bool


@app.post("/api/watching")
def set_watching(body: Watching, x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    _state["watching"] = body.on
    return {"watching": body.on}


@app.post("/api/frame")
async def frame(frames: list[UploadFile] = File(...), x_lastseen_passcode: str | None = Header(None)):
    """One watching moment. The page sends 1-3 quick frames every few seconds; only the
    sharpest is used (a handheld phone blurs most frames in dim light; picking is free)."""
    _check_passcode(x_lastseen_passcode)
    if not _state["watching"]:
        raise HTTPException(409, "Watching is off.")
    imgs = await _images(frames, 1, 3)
    scores = [sharpness(i) for i in imgs]
    best = imgs[scores.index(max(scores))]
    result = await _run(_pipeline().process, best)
    result["spent"] = _spent()
    result["sharpness"] = [round(s, 1) for s in scores]
    _log_watch(best, result)
    return result


def sharpness(img: np.ndarray) -> float:
    from .meta_client import prepare_frame
    g = cv2.cvtColor(prepare_frame(img), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


LIVE_DIR = HOME_DIR / "live"


def _log_watch(img: np.ndarray, result: dict) -> None:
    """While testing: one line per frame in the terminal, and (unless
    LASTSEEN_SAVE_FRAMES=0) the frame + result in home/live/ so we can replay it.
    Local only, git-ignored; delete home/live/ when done."""
    if result.get("skipped"):
        why = f"skipped: {result['skipped']}"
    elif not result.get("candidates"):
        why = "processed: SAM found no enrolled item"
    else:
        parts = []
        for c in result["candidates"]:
            sp = c.get("spark") or {}
            parts.append(f"{c['item']} box={c['box']} spark_match={sp.get('is_users_item')} "
                         f"conf={sp.get('confidence')} landmark={sp.get('landmark')} rel={sp.get('relation')}")
        st = [f"{s['item']}:{s['status']}" for s in result.get("sightings", [])]
        why = "processed: " + " | ".join(parts) + f" -> stored {st or 'nothing'}"
        if result.get("abstained"):
            why += f" abstained={result['abstained']}"
    if result.get("sharpness"):
        why += f" (sharpness of {len(result['sharpness'])} frames: {result['sharpness']})"
    print(f"[watch] {time.strftime('%H:%M:%S')} {why}", flush=True)
    if os.environ.get("LASTSEEN_SAVE_FRAMES", "1") == "0" or result.get("skipped") == "unchanged":
        return
    d = LIVE_DIR / time.strftime("%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "frame.jpg"), img)
    (d / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k != "calls"}, default=str, indent=1))


def _where(item_id: str | None) -> dict:
    home = Home.load()
    if not home.items:
        return {"text": "No items enrolled yet.", "status": "no_items"}
    if item_id is None:
        names = ", ".join(i.name for i in home.items)
        return {"text": f"Which one? I keep track of: {names}.", "status": "which"}
    return answer_where(item_id, MemoryStore().for_item(item_id), home)


class Ask(BaseModel):
    text: str


@app.post("/api/ask")
def ask(body: Ask, x_lastseen_passcode: str | None = Header(None)):
    """Typed question (same rules as voice)."""
    _check_passcode(x_lastseen_passcode)
    return _handle(body.text)


def _handle(text: str) -> dict:
    home = Home.load()
    intent, item_id = voice.parse_intent(text, home.item_names())
    if intent == "where":
        out = _where(item_id)
    elif intent == "guide":
        out = ({"text": f"Which one? I keep track of: {', '.join(i.name for i in home.items)}.", "status": "which"}
               if item_id is None else {"text": "Starting the search.", "status": "guide"})
    elif intent == "watch_on":
        _state["watching"] = True
        out = {"text": "Watching. I'll remember where I see your things.", "status": "watch_on"}
    elif intent == "watch_off":
        _state["watching"] = False
        out = {"text": "Stopped watching.", "status": "watch_off"}
    elif intent == "forget":
        n = MemoryStore().forget()
        out = {"text": f"Forgotten. Deleted {n} memories.", "status": "forget"}
    elif intent == "stop":
        out = {"text": "OK.", "status": "stop"}
    else:
        out = {"text": voice.HELP, "status": "help"}
    out.update({"transcript": text, "intent": intent, "item": item_id})
    s = out.pop("sighting", None)
    if s is not None:
        out["sighting"] = s.__dict__
    return out


@app.post("/api/voice")
async def ask_voice(audio: UploadFile = File(...), x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    wav = await audio.read()
    names = [n for ns in Home.load().item_names().values() for n in ns]
    try:
        tr = await asyncio.to_thread(voice.transcribe, _client(), wav, None, "voice", names)
    except voice.VoiceError as e:
        raise HTTPException(422, str(e))
    except (BudgetExceeded, UnpricedModel) as e:
        raise HTTPException(402, f"Budget: {e}")
    return {**_handle(tr["transcript"]), "seconds": tr["seconds"], "spent": _spent()}


@app.post("/api/find")
async def find(frames: list[UploadFile] = File(...), item: str = Form(...),
               x_lastseen_passcode: str | None = Header(None)):
    """Final step of a search: is the item in this view, and which way?"""
    _check_passcode(x_lastseen_passcode)
    try:
        Home.load().item(item)
    except KeyError:
        raise HTTPException(404, f"No enrolled item '{item}'.")
    imgs = await _images(frames, 1, 1)
    out = await _run(_pipeline().find, imgs[0], item)
    out["spent"] = _spent()
    return out


@app.get("/api/memories")
def memories(x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    return {"memories": [s.__dict__ for s in MemoryStore().all()][-50:]}


@app.post("/api/forget")
def forget(item: str | None = None, x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    return {"deleted": MemoryStore().forget(item)}


# ---------------------------------------------------------------- guide me to it (Muse Wonder)

_guides: dict[str, guide_mod.Session] = {}


class GuideStart(BaseModel):
    item: str
    said: str = ""


def _guide_reply(g: guide_mod.Session, step: guide_mod.Step, look: dict | None = None) -> dict:
    out = {"session": g.id, "action": step.action, "say": step.say, "place": step.place,
           "found": step.found, "planner": g.planner, "trace": g.trace, "look": look, "spent": _spent()}
    if step.action == "done":
        _guides.pop(g.id, None)
    return out


@app.post("/api/guide/start")
async def guide_start(body: GuideStart, x_lastseen_passcode: str | None = Header(None)):
    _check_passcode(x_lastseen_passcode)
    home = Home.load()
    try:
        home.item(body.item)
    except KeyError:
        raise HTTPException(404, f"No enrolled item '{body.item}'.")
    g, step = await _run(guide_mod.start, _client(), body.item, home, MemoryStore().for_item(body.item), body.said)
    _guides.clear()                       # one search at a time
    _guides[g.id] = g
    return _guide_reply(g, step)


def _guide_view(g: guide_mod.Session, img: np.ndarray) -> dict:
    place = g.pending
    look = _pipeline().find(img, g.item, None if place == guide_mod.HERE else place)
    step = guide_mod.add_view(_client(), g, look)
    if step is None:
        return {"session": g.id, "action": "look", "say": None, "place": place, "found": False,
                "planner": g.planner, "trace": g.trace, "look": look, "spent": _spent()}
    return _guide_reply(g, step, look)


@app.post("/api/guide/frame")
async def guide_frame(session: str = Form(...), frames: list[UploadFile] = File(...),
                      x_lastseen_passcode: str | None = Header(None)):
    """One camera view during a look. Keep sending views while action == "look"."""
    _check_passcode(x_lastseen_passcode)
    g = _guides.get(session)
    if g is None or g.pending is None:
        raise HTTPException(409, "That search has ended. Say: guide me to my glasses.")
    imgs = await _images(frames, 1, 1)
    return await _run(_guide_view, g, imgs[0])
