"""Voice in: WAV checks, the Muse Voice Transcribe call (faked), cost, and intent rules."""
import io
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from holdpoint.budget import Ledger, Price
from holdpoint.meta_client import MetaClient
from holdpoint.voice import VoiceError, parse_intent, transcribe, wav_seconds


def wav(seconds=2.0, rate=16000, channels=1, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate) * channels, dtype=np.int16 if width == 2 else np.uint8).tobytes())
    return buf.getvalue()


def client(tmp_path):
    ledger = Ledger(cap_usd=1.0, ledger_file=tmp_path / "ledger.jsonl",
                    prices={"muse-voice-transcribe-1.0": Price(per_audio_hour=0.18)})
    return MetaClient(ledger=ledger, client=object(), cache_dir=tmp_path / "cache")


def test_wav_checks():
    assert wav_seconds(wav(2.0)) == pytest.approx(2.0)
    assert wav_seconds(wav(1.0, rate=24000)) == pytest.approx(1.0)
    for bad in [wav(2.0, channels=2), wav(2.0, rate=44100), wav(0.1), wav(20.0), b"not audio"]:
        with pytest.raises(VoiceError):
            wav_seconds(bad)


def test_transcribe_sends_wav_and_records_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    seen = {}

    def fake_post(url, headers, files, timeout):
        seen.update(url=url, headers=headers, request=json.loads(files["request"][1]), audio=files["audio"])
        return SimpleNamespace(status_code=200, json=lambda: {"transcript": " Are the burners off? "})

    c = client(tmp_path)
    out = transcribe(c, wav(5.0), post=fake_post)
    assert out == {"transcript": "Are the burners off?", "seconds": 5.0}
    assert seen["url"].endswith("/v1/asr/transcribe") and seen["headers"]["Authorization"] == "Bearer k"
    assert seen["request"]["model"] == "muse-voice-transcribe-1.0" and seen["request"]["audioEncoding"] == "WAV"
    assert seen["audio"][2] == "audio/wav"
    assert c.ledger.spent() == pytest.approx(5 / 3600 * 0.18)


def test_transcribe_error_is_reported_and_still_billed(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    c = client(tmp_path)
    with pytest.raises(VoiceError):
        transcribe(c, wav(2.0), post=lambda *a, **k: SimpleNamespace(status_code=400, text="Unsupported WAV"))
    assert c.ledger.spent() > 0


@pytest.mark.parametrize("text,intent", [
    ("Hold Point, are the burners off?", "check_stove"),
    ("check the stove", "check_stove"),
    ("Is the stove on", "check_stove"),
    ("I'm leaving, check the house", "leaving"),
    ("I'm heading out", "leaving"),
    ("What does the oven display say", "check_oven"),
    ("what's the weather", None),
    ("", None),
])
def test_intents(text, intent):
    assert parse_intent(text) == intent


def test_voice_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from holdpoint import server
    monkeypatch.setenv("HOLDPOINT_PASSCODE", "secret")
    monkeypatch.setattr(server, "_client", lambda: client(tmp_path))
    monkeypatch.setattr(server, "transcribe", lambda c, data: {"transcript": "Are the burners off?", "seconds": 2.0})
    http = TestClient(server.app)
    files = {"audio": ("command.wav", wav(2.0), "audio/wav")}
    assert http.post("/api/voice", files=files).status_code == 401
    r = http.post("/api/voice", files=files, headers={"X-Holdpoint-Passcode": "secret"}).json()
    assert r["intent"] == "check_stove" and r["action"] == "check"
