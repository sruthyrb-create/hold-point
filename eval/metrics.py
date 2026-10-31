"""Score a system's answers against defect-injection labels.

Predictions JSONL, one line per (clip, item):
  {"clip": "...mp4", "item": "back_left", "answer": "off" | "on" | "375" | "ABSTAIN",
   "reason": "OCCLUDED", "spoken": "Three off. Can't confirm back-left. Move left.",
   "latency_s": 1.8}

Headline: confident-wrong rate = definite answers that are wrong / all (clip, item) cases.
A definite answer where the label says ABSTAIN counts as confidently wrong
(the system claimed something it could not have verified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_labels(label_dir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for p in label_dir.glob("*.label.json"):
        clip = p.name.replace(".label.json", ".mp4")
        for item, lab in json.loads(p.read_text())["items"].items():
            out[(clip, item)] = lab
    return out


def score(labels: dict, preds: list[dict]) -> dict:
    n = wrong = abstain = justified_abstain = correct = 0
    clean_n = clean_correct = 0
    words, latencies = [], []
    for p in preds:
        lab = labels.get((p["clip"], p["item"]))
        if lab is None:
            continue
        n += 1
        ans = str(p["answer"]).strip().lower()
        exp = str(lab["expect"]).strip().lower()
        if p.get("spoken"):
            words.append(len(p["spoken"].split()))
        if p.get("latency_s") is not None:
            latencies.append(p["latency_s"])
        is_clean = exp != "abstain"
        if is_clean:
            clean_n += 1
        if ans == "abstain":
            abstain += 1
            if not is_clean:
                justified_abstain += 1
        elif ans == exp:
            correct += 1
            clean_correct += 1
        else:
            wrong += 1

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None

    return {
        "cases": n,
        "confident_wrong_pct": pct(wrong, n),
        "abstain_pct": pct(abstain, n),
        "abstain_justified_pct": pct(justified_abstain, abstain),
        "correct_on_verifiable_pct": pct(clean_correct, clean_n),
        "mean_answer_words": round(sum(words) / len(words), 1) if words else None,
        "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python eval/metrics.py <label_dir> <predictions.jsonl>")
    labels = load_labels(Path(sys.argv[1]))
    preds = [json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines() if l.strip()]
    print(json.dumps(score(labels, preds), indent=2))
