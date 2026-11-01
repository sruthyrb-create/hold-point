"""Check the Muse Voice Transcribe call works before trying it on the phone.

    python scripts/voice_smoke.py              # sends 2 s of silence (cost ~ $0.0001)
    python scripts/voice_smoke.py speech.wav   # or your own mono 16-bit 16/24 kHz WAV

Success = "OK" plus the transcript (empty for silence).
"""
from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint.meta_client import MetaClient  # noqa: E402
from holdpoint.voice import transcribe  # noqa: E402


def silence(seconds: float = 2.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buf.getvalue()


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    data = Path(sys.argv[1]).read_bytes() if len(sys.argv) > 1 else silence()
    client = MetaClient()
    out = transcribe(client, data, tag="smoke")
    print(f"OK  {out['seconds']} s of audio  transcript: {out['transcript']!r}")
    print(client.ledger.summary())
