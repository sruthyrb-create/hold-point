"""Voice in: Muse Voice Transcribe turns a short spoken command into text, then a
plain keyword rule (no AI) decides what to do.

The phone records a few seconds of speech, encodes it as WAV (mono, 16-bit PCM,
16 kHz) and the server sends it to Meta's file endpoint:

    POST https://api.meta.ai/v1/asr/transcribe   (multipart: "request" JSON + "audio" WAV)

Cost: $0.18 per audio hour, i.e. about $0.0003 for a 5-second command.
"""
from __future__ import annotations

import io
import json
import os
import re
import wave
from typing import Callable

MODEL = "muse-voice-transcribe-1.0"
URL = "https://api.meta.ai/v1/asr/transcribe"
MAX_SECONDS = 15
KEYWORDS = ["Hold Point", "stove", "burner", "burners", "knob", "knobs", "oven", "off"]


class VoiceError(ValueError):
    pass


def wav_seconds(wav_bytes: bytes) -> float:
    """Check the WAV is what Meta accepts and return its length in seconds."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    except (wave.Error, EOFError) as e:
        raise VoiceError(f"Not a WAV recording ({e}).")
    if channels != 1 or width != 2 or rate not in (16000, 24000):
        raise VoiceError(f"Recording must be mono 16-bit 16/24 kHz (got {channels} ch, {8 * width}-bit, {rate} Hz).")
    seconds = frames / rate
    if seconds < 0.3:
        raise VoiceError("Recording was too short. Tap, speak, then tap again.")
    if seconds > MAX_SECONDS:
        raise VoiceError(f"Recording was longer than {MAX_SECONDS} seconds.")
    return seconds


def transcribe(client, wav_bytes: bytes, post: Callable | None = None, tag: str = "voice") -> dict:
    """Return {"transcript": str, "seconds": float}. `client` is a MetaClient (for its ledger)."""
    seconds = wav_seconds(wav_bytes)
    ledger = client.ledger
    ledger.check(MODEL, expected_cost=ledger.unit_cost(MODEL, audio_seconds=seconds))
    request = {"model": MODEL, "mode": "PUSH_TO_TALK", "audioEncoding": "WAV",
               "languageBias": ["English"], "keywords": KEYWORDS}
    if post is None:
        import httpx
        post = httpx.post
    resp = post(URL, headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}",
                              "Accept": "application/json"},
                files={"request": (None, json.dumps(request), "application/json"),
                       "audio": ("command.wav", wav_bytes, "audio/wav")},
                timeout=30)
    ledger.record_units(MODEL, audio_seconds=seconds, tag=tag)   # billed once sent, whatever the outcome
    if resp.status_code != 200:
        raise VoiceError(f"Transcription failed ({resp.status_code}): {resp.text[:150]}")
    return {"transcript": (resp.json().get("transcript") or "").strip(), "seconds": round(seconds, 2)}


# ---------------------------------------------------------------- what did the user ask?

LEAVING = re.compile(r"\b(leav(e|ing)|going out|head(ing)? out|check (the|my) (house|home|place)|lock(ing)? up)\b")
STOVE = re.compile(r"\b(stove|burners?|knobs?|hob|cooktop|cook top)\b")
OVEN = re.compile(r"\b(oven|display|temperature)\b")
CHECK = re.compile(r"\b(check|are|is|did|off|on)\b")


def parse_intent(text: str) -> str | None:
    """Map a transcript to one of: "leaving", "check_stove", "check_oven", or None.

    Deliberately a plain rule, not a model: what Hold Point does next must be
    predictable, and a wrong guess here only costs a "Say: check the stove".
    """
    t = text.lower()
    if LEAVING.search(t):
        return "leaving"
    if OVEN.search(t):
        return "check_oven"
    if STOVE.search(t) and CHECK.search(t):
        return "check_stove"
    if STOVE.search(t):
        return "check_stove"
    return None


HELP = "Say: check the stove. Or: I'm leaving, check the house."
