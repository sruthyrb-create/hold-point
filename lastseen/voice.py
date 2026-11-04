"""Voice in: Muse Voice Transcribe turns a short spoken question into text, then a
plain rule (no AI) decides what to do. Transcription code copied from Hold Point.

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
KEYWORDS = ["Last Seen", "where", "glasses", "keys", "pen", "wallet", "phone"]  # item names are added per call


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


def transcribe(client, wav_bytes: bytes, post: Callable | None = None, tag: str = "voice",
               extra_keywords: list[str] | None = None) -> dict:
    """Return {"transcript": str, "seconds": float}. `client` is a MetaClient (for its ledger)."""
    seconds = wav_seconds(wav_bytes)
    ledger = client.ledger
    ledger.check(MODEL, expected_cost=ledger.unit_cost(MODEL, audio_seconds=seconds))
    request = {"model": MODEL, "mode": "PUSH_TO_TALK", "audioEncoding": "WAV",
               "languageBias": ["English"], "keywords": KEYWORDS + (extra_keywords or [])}
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

STOP = re.compile(r"\b(stop|cancel|never ?mind|be quiet)\b")
WATCH_ON = re.compile(r"\b(start|resume) (watching|remembering)\b")
WATCH_OFF = re.compile(r"\b(stop|pause) (watching|remembering)\b")
FORGET = re.compile(r"\b(forget|delete|erase)\b")
FIND = re.compile(r"\b(where|find|lost|left|seen|see)\b")
GUIDE = re.compile(r"\b(guide me|take me|help me find|lead me|walk me)\b")


def parse_intent(text: str, item_names: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """Map a transcript to (intent, item_id).

    intents: "where" / "guide" (item_id set), "watch_on", "watch_off", "forget", "stop",
    or (None, None). item_names: {"glasses": ["glasses", "specs"], ...}.
    A plain rule, not a model, so what happens next is predictable.
    """
    t = text.lower()
    if WATCH_OFF.search(t):
        return "watch_off", None
    if WATCH_ON.search(t):
        return "watch_on", None
    if FORGET.search(t):
        return "forget", None
    if STOP.search(t):
        return "stop", None
    def variants(n: str) -> set[str]:                 # "keys" also matches "key", "wallet" also "wallets"
        n = n.lower()
        return {n, n[:-1]} if n.endswith("s") and len(n) > 3 else {n, n + "s"}
    found = [iid for iid, names in item_names.items()
             if any(re.search(rf"\b{re.escape(v)}\b", t) for n in names for v in variants(n))]
    if len(found) == 1:
        return ("guide" if GUIDE.search(t) else "where"), found[0]
    if GUIDE.search(t) and not found:
        return "guide", None
    if FIND.search(t) and not found:
        return "where", None          # asked, but which item is unclear -> app asks back
    return None, None


HELP = "Say: where are my glasses? Or: guide me to my glasses."
