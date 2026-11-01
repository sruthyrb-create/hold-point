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
from .decision import DecisionConfig
from .profile import HOME_DIR, load_stove
from .setup_live import SetupFailed, setup_from_frames
from .voice import HELP, VoiceError, parse_intent, transcribe

STATIC = Path(__file__).resolve().parent / "static"
MAX_FRAMES = 6
CFG = DecisionConfig(min_visible_frames=3)

app = FastAPI(title="Hold Point")
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
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status(x_holdpoint_passcode: str | None = Header(None)):
    _check_passcode(x_holdpoint_passcode)
    stove, _ = _stove()
    ledger = _client().ledger
    return {"knobs": [k["name"] for k in stove["knobs"]], "spent_usd": round(ledger.spent(), 4),
            "cap_usd": ledger.cap}


@app.post("/api/check")
async def check(frames: list[UploadFile] = File(...), x_holdpoint_passcode: str | None = Header(None)):
    _check_passcode(x_holdpoint_passcode)
    if _busy.locked():
        raise HTTPException(429, "A check is already running.")
    async with _busy:
        images = await _read_images(frames)
        stove, refs = _stove()
        cfg = CFG if len(images) >= CFG.min_visible_frames else DecisionConfig(min_visible_frames=len(images))
        t = time.perf_counter()
        try:
            result = await run_in_threadpool(run_stove_check, _client(), images, stove, refs, cfg)
        except BudgetExceeded as e:
            raise HTTPException(402, f"Budget cap reached: {e}")
        result["record"] = log_check(result)
        result["server_seconds"] = round(time.perf_counter() - t, 2)
        save_frames(images, result)
        return result


# What the phone does after each spoken command. "check" = capture frames and run a stove check.
INTENT_REPLY = {
    "check_stove": ("check", "Checking the stove. Hold still."),
    "leaving": ("check", "Okay. Checking the stove first. Hold still."),
    "check_oven": (None, "Oven display check isn't ready yet. Say: check the stove."),
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
