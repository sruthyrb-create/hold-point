"""Spend guard: every API call is priced, logged, and refused once the cap is hit.

Prices live in prices.json, copied from the Meta dashboard. Token models are
priced per 1M tokens; SAM 3.1 per image or per video frame; Voice Transcribe
per audio hour. A model not listed there is BLOCKED.

Cached input is billed at full input price here (conservative). The Billing
page is the source of truth, so compare the two after your first few calls.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICES_FILE = ROOT / "prices.json"
LEDGER_FILE = ROOT / ".holdpoint" / "ledger.jsonl"
DEFAULT_CAP_USD = float(os.environ.get("HOLDPOINT_BUDGET_USD", "120"))


class BudgetExceeded(RuntimeError):
    pass


class UnpricedModel(RuntimeError):
    pass


@dataclass
class Price:
    input_per_m: float | None = None
    output_per_m: float | None = None
    per_image: float | None = None
    per_video_frame: float | None = None
    per_audio_hour: float | None = None


def load_prices(path: Path = PRICES_FILE) -> dict[str, Price]:
    raw = json.loads(path.read_text())
    fields = Price.__dataclass_fields__
    out = {}
    for model, p in raw.items():
        if model.startswith("_"):
            continue
        price = Price(**{k: float(v) for k, v in p.items() if k in fields and v is not None})
        if any(getattr(price, f) is not None for f in fields):
            out[model] = price
    return out


class Ledger:
    def __init__(self, cap_usd: float = DEFAULT_CAP_USD, ledger_file: Path = LEDGER_FILE,
                 prices: dict[str, Price] | None = None):
        self.cap = cap_usd
        self.file = ledger_file
        self.prices = prices if prices is not None else load_prices()
        self.file.parent.mkdir(parents=True, exist_ok=True)

    def spent(self) -> float:
        if not self.file.exists():
            return 0.0
        return sum(json.loads(l)["cost_usd"] for l in self.file.read_text().splitlines() if l.strip())

    def check(self, model: str, expected_cost: float = 0.0) -> None:
        """Call BEFORE every request. expected_cost lets unit-priced calls (e.g. a
        500-frame SAM video job) be refused if they would cross the cap."""
        if model not in self.prices:
            raise UnpricedModel(f"No price for '{model}' in prices.json. Add it from the dashboard first.")
        spent = self.spent()
        if spent + expected_cost > self.cap or spent >= self.cap:
            raise BudgetExceeded(
                f"Spent ${spent:.2f} of ${self.cap:.2f} cap (this call ~${expected_cost:.4f}). "
                "Raise HOLDPOINT_BUDGET_USD to continue.")

    def _write(self, entry: dict) -> float:
        entry["t"] = time.time()
        with self.file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry["cost_usd"]

    def record(self, model: str, input_tokens: int, output_tokens: int, tag: str = "") -> float:
        p = self.prices[model]
        if p.input_per_m is None or p.output_per_m is None:
            raise UnpricedModel(f"'{model}' has no token price.")
        cost = input_tokens / 1e6 * p.input_per_m + output_tokens / 1e6 * p.output_per_m
        return self._write({"model": model, "in": input_tokens, "out": output_tokens,
                            "cost_usd": round(cost, 6), "tag": tag})

    def unit_cost(self, model: str, images: int = 0, video_frames: int = 0, audio_seconds: float = 0) -> float:
        p = self.prices[model]
        cost = 0.0
        for n, rate, name in [(images, p.per_image, "per_image"), (video_frames, p.per_video_frame, "per_video_frame"),
                              (audio_seconds / 3600, p.per_audio_hour, "per_audio_hour")]:
            if n:
                if rate is None:
                    raise UnpricedModel(f"'{model}' has no {name} price.")
                cost += n * rate
        return cost

    def record_units(self, model: str, images: int = 0, video_frames: int = 0,
                     audio_seconds: float = 0, tag: str = "") -> float:
        cost = self.unit_cost(model, images, video_frames, audio_seconds)
        return self._write({"model": model, "images": images, "video_frames": video_frames,
                            "audio_s": audio_seconds, "cost_usd": round(cost, 6), "tag": tag})

    def summary(self) -> str:
        return f"${self.spent():.4f} spent of ${self.cap:.2f} cap"
