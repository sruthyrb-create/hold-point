"""Scoring rules for the live evaluation."""
from eval.live_eval import DEFECTS, score_baseline, score_holdpoint, summarise
import numpy as np


def hp(all_clear, confirmed_on=(), errors=()):
    items = [{"id": f"knob_{i}", "verdict": {"answered": f"knob_{i}" in confirmed_on or all_clear,
                                              "state": "on" if f"knob_{i}" in confirmed_on else "off"}}
             for i in range(1, 5)]
    return {"all_clear": all_clear, "items": items, "errors": list(errors)}


def test_holdpoint_buckets():
    assert score_holdpoint(hp(True), "all_off") == "correct"
    assert score_holdpoint(hp(False), "all_off") == "abstain"
    assert score_holdpoint(hp(False, ["knob_2"]), "all_off") == "false_alarm"
    assert score_holdpoint(hp(True), "on:knob_3") == "unsafe"
    assert score_holdpoint(hp(False, ["knob_3"]), "on:knob_3") == "correct"
    assert score_holdpoint(hp(False, ["knob_1"]), "slight:knob_3") == "wrong_knob"
    assert score_holdpoint(hp(False), "slight:knob_1") == "abstain"
    assert score_holdpoint(hp(True, errors=["x"]), "all_off") == "abstain"


def test_baseline_buckets():
    assert score_baseline({"all_off": True, "knobs_on": []}, "all_off") == "correct"
    assert score_baseline({"all_off": False, "knobs_on": ["knob_1"]}, "all_off") == "false_alarm"
    assert score_baseline({"all_off": True, "knobs_on": []}, "on:knob_3") == "unsafe"
    assert score_baseline({"all_off": False, "knobs_on": ["knob_3"]}, "on:knob_3") == "correct"
    assert score_baseline({"all_off": False, "knobs_on": ["knob_1", "knob_2", "knob_3", "knob_4"]}, "on:knob_3") == "wrong_knob"
    assert score_baseline({"all_off": False, "knobs_on": ["knob_4"]}, "on:knob_3") == "wrong_knob"
    assert score_baseline({"all_off": None, "knobs_on": []}, "on:knob_3") == "error"


def test_summary_rates():
    rows = [{"system": "baseline", "condition": "clean", "truth": "on:knob_3", "bucket": "unsafe"},
            {"system": "baseline", "condition": "clean", "truth": "all_off", "bucket": "correct"},
            {"system": "baseline", "condition": "clean", "truth": "all_off", "bucket": "error"},
            {"system": "holdpoint", "condition": "clean", "truth": "on:knob_3", "bucket": "abstain"}]
    b = summarise(rows, "baseline")
    assert b["n"] == 2 and b["unsafe_rate_when_on"] == 1.0 and b["confidently_wrong_rate"] == 0.5
    h = summarise(rows, "holdpoint")
    assert h["abstain_rate"] == 1.0 and h["unsafe_rate_when_on"] == 0.0


def test_defects_keep_shape():
    f = np.full((120, 160, 3), 128, np.uint8)
    for fn in DEFECTS.values():
        out = fn([f])
        assert out[0].shape == f.shape
