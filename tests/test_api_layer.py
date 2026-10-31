"""Tests for the budget guard, cache and Witness A, using a fake API (no key, no cost)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from holdpoint.budget import BudgetExceeded, Ledger, Price, UnpricedModel
from holdpoint.meta_client import MetaClient
from holdpoint.witness_vlm import read_frames


class FakeOpenAI:
    def __init__(self, reply: dict, prompt_tokens=2000, completion_tokens=300):
        self.calls = 0
        self.reply, self.pt, self.ct = reply, prompt_tokens, completion_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=self.pt, completion_tokens=self.ct),
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.reply)))])


PRICES = {"muse-spark-1.3": Price(1.25, 4.25)}
REPLY = {"frames": [
    {"frame": 0, "items": [
        {"item_id": "bl", "visible": True, "state": "off", "confidence": 0.9, "reason_code": "OK"},
        {"item_id": "invented", "visible": True, "state": "on", "confidence": 1.0, "reason_code": "OK"}]},
    {"frame": 1, "items": [
        {"item_id": "bl", "visible": False, "state": None, "confidence": 0.0, "reason_code": "OCCLUDED"}]},
    {"frame": 7, "items": []},  # out-of-range frame number: ignored
]}


def make(tmp_path, cap=1.0, reply=REPLY):
    fake = FakeOpenAI(reply)
    ledger = Ledger(cap_usd=cap, ledger_file=tmp_path / "ledger.jsonl", prices=PRICES)
    return MetaClient(ledger=ledger, client=fake, cache_dir=tmp_path / "cache"), fake, ledger


def frames(n):
    return [np.full((720, 1280, 3), 128, np.uint8) for _ in range(n)]


def test_readings_parsed_and_invented_items_dropped(tmp_path):
    client, fake, _ = make(tmp_path)
    out = read_frames(client, frames(2), [10, 11], {"bl": "back-left knob"})
    assert [f.frame_index for f in out] == [10, 11]
    assert [r.item_id for r in out[0].items] == ["bl"]
    assert out[1].items[0].reason_code.value == "OCCLUDED"
    assert fake.last["reasoning_effort"] == "low"


def test_cache_makes_repeat_calls_free(tmp_path):
    client, fake, ledger = make(tmp_path)
    read_frames(client, frames(2), [0, 1], {"bl": "knob"})
    spent = ledger.spent()
    read_frames(client, frames(2), [0, 1], {"bl": "knob"})
    assert fake.calls == 1 and ledger.spent() == spent
    assert spent == pytest.approx(2000 / 1e6 * 1.25 + 300 / 1e6 * 4.25)


def test_cap_blocks_calls(tmp_path):
    client, fake, ledger = make(tmp_path, cap=0.0)
    with pytest.raises(BudgetExceeded):
        read_frames(client, frames(1), [0], {"bl": "knob"})
    assert fake.calls == 0


def test_unit_pricing_from_real_price_file(tmp_path):
    ledger = Ledger(cap_usd=1.0, ledger_file=tmp_path / "l.jsonl")  # reads prices.json
    assert ledger.unit_cost("sam-3.1", images=1000) == pytest.approx(2.50)
    assert ledger.unit_cost("sam-3.1", video_frames=1000) == pytest.approx(0.20)
    assert ledger.unit_cost("muse-voice-transcribe-1.0", audio_seconds=3600) == pytest.approx(0.18)
    ledger.record_units("sam-3.1", video_frames=500)
    assert ledger.spent() == pytest.approx(0.10)
    with pytest.raises(BudgetExceeded):  # a job that would cross the cap is refused up front
        ledger.check("sam-3.1", expected_cost=ledger.unit_cost("sam-3.1", images=400))


def test_unpriced_model_blocked(tmp_path):
    ledger = Ledger(cap_usd=10, ledger_file=tmp_path / "l.jsonl", prices=PRICES)
    with pytest.raises(UnpricedModel):
        ledger.check("sam-3.1")
