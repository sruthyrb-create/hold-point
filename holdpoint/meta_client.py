"""The only place that talks to the Meta Model API.

Every call goes: cache lookup -> budget check -> request -> record cost -> cache save.
A repeated identical request (same model, prompt, images, settings) costs nothing.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .budget import Ledger

BASE_URL = "https://api.meta.ai/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".holdpoint" / "cache"
MAX_SIDE = 1280  # 720p frames stay as they are; anything larger is shrunk


def encode_frame(frame_bgr: np.ndarray, quality: int = 85) -> str:
    h, w = frame_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not JPEG-encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class MetaClient:
    def __init__(self, ledger: Ledger | None = None, client: Any = None, cache_dir: Path = CACHE_DIR):
        self.ledger = ledger or Ledger()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if client is None:
            from openai import OpenAI  # imported lazily so tests don't need a key
            client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
        self.client = client

    def _key(self, payload: dict) -> Path:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def chat_json(self, model: str, system: str, text: str, images: list[str],
                  schema: dict, reasoning_effort: str = "low", tag: str = "") -> dict:
        """Chat Completions call with images + structured output. Returns parsed JSON."""
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in images]
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "reasoning_effort": reasoning_effort,
        }
        cached = self._key(payload)
        if cached.exists():
            return json.loads(cached.read_text())["result"]

        self.ledger.check(model)
        resp = self.client.chat.completions.create(**payload)
        usage = resp.usage
        self.ledger.record(model, usage.prompt_tokens, usage.completion_tokens, tag=tag)
        result = json.loads(resp.choices[0].message.content)
        cached.write_text(json.dumps({"result": result, "usage": {
            "in": usage.prompt_tokens, "out": usage.completion_tokens}}))
        return result
