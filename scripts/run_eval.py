"""Evaluation: plain Muse Spark vs Hold Point on your labelled live checks.

    python scripts/run_eval.py                  # clean frames only (cheap: ~0.3 cents per check)
    python scripts/run_eval.py --defects        # + dark / blur / far / glare copies (~2.5 cents each)
    python scripts/run_eval.py --defects --limit 10

Uses every home/live/<time>/ folder whose result.json has a "truth" label (set with
Evaluation mode on the phone). Results are saved to eval/results/rows.jsonl, and a run
can be stopped and restarted: finished rows are skipped. The summary is printed and
written to eval/results/summary.md for the README.

Hold Point on clean frames = the answer it actually gave live (nothing re-run).
Baseline = plain Muse Spark, frame 1, whole picture, one question, must answer yes/no.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.live_eval import DEFECTS, score_baseline, score_holdpoint, summarise  # noqa: E402
from holdpoint.check import run_stove_check  # noqa: E402
from holdpoint.decision import DecisionConfig  # noqa: E402
from holdpoint.meta_client import MetaClient, encode_frame  # noqa: E402
from holdpoint.profile import load_stove  # noqa: E402

OUT = ROOT / "eval" / "results"
BASELINE_MODEL = "muse-spark-1.3"
BASELINE_PROMPT = ("This is a photo of a stove's control knobs. The knobs are numbered knob_1 to knob_{n} "
                   "from left to right. Are ALL the burner knobs turned OFF? Answer with all_off true or "
                   "false, and list in knobs_on the ids of any knob that is not OFF.")
BASELINE_SCHEMA = {"name": "Baseline", "schema": {
    "type": "object", "additionalProperties": False, "required": ["all_off", "knobs_on"],
    "properties": {"all_off": {"type": "boolean"}, "knobs_on": {"type": "array", "items": {"type": "string"}}}}}


def baseline(client, frame, n_knobs: int) -> dict:
    return client.chat_json(BASELINE_MODEL, "You answer questions about photos.",
                            BASELINE_PROMPT.format(n=n_knobs), [encode_frame(frame)],
                            BASELINE_SCHEMA, reasoning_effort="low", tag="eval-baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=str(ROOT / "home" / "live"))
    ap.add_argument("--defects", action="store_true", help="also test damaged copies (costs more)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N labelled checks")
    a = ap.parse_args()

    cases = []
    for d in sorted(Path(a.live).iterdir()):
        f = d / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("truth") and not r["truth"].startswith("oven"):   # stove checks only
                cases.append((d, r))
    if a.limit:
        cases = cases[: a.limit]
    if not cases:
        raise SystemExit("No labelled checks yet. Use Evaluation mode on the phone first.")

    OUT.mkdir(parents=True, exist_ok=True)
    rows_file = OUT / "rows.jsonl"
    done = set()
    rows = []
    if rows_file.exists():
        for line in rows_file.read_text().splitlines():
            row = json.loads(line)
            rows.append(row)
            done.add((row["case"], row["system"], row["condition"]))

    stove, refs = load_stove()
    n_knobs = len(stove["knobs"])
    client = MetaClient()
    conditions = ["clean"] + (list(DEFECTS) if a.defects else [])
    est = len(cases) * (0.003 + (0.025 + 0.003) * (len(conditions) - 1))
    print(f"{len(cases)} labelled checks, conditions: {', '.join(conditions)}. Estimated cost up to ${est:.2f}.")

    def add(row):
        rows.append(row)
        with rows_file.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    for i, (d, saved) in enumerate(cases, 1):
        truth = saved["truth"]
        frames = [cv2.imread(str(p)) for p in sorted(d.glob("frame*.jpg"))]
        for cond in conditions:
            imgs = frames if cond == "clean" else DEFECTS[cond](frames)
            if (d.name, "holdpoint", cond) not in done:
                hp = saved if cond == "clean" else run_stove_check(
                    client, imgs, stove, refs, DecisionConfig(min_visible_frames=3), tag="eval")
                add({"case": d.name, "truth": truth, "condition": cond, "system": "holdpoint",
                     "answer": hp["answer"], "bucket": score_holdpoint(hp, truth)})
            if (d.name, "baseline", cond) not in done:
                try:
                    rep = baseline(client, imgs[0], n_knobs)
                except Exception as e:   # a failed baseline call is recorded, not hidden
                    rep = {"all_off": None, "knobs_on": [], "error": str(e)[:120]}
                add({"case": d.name, "truth": truth, "condition": cond, "system": "baseline",
                     "answer": json.dumps(rep), "bucket": score_baseline(rep, truth)})
        print(f"  [{i}/{len(cases)}] {d.name} ({truth}) done")

    lines = ["| Condition | System | Checks | Correct | Unsafe 'all off' | False alarm | Wrong knob | Abstained |",
             "|---|---|---|---|---|---|---|---|"]
    for cond in conditions + [None]:
        for system, label in [("baseline", "Plain Muse Spark"), ("holdpoint", "Hold Point")]:
            s = summarise(rows, system, cond)
            if not s["n"]:
                continue
            lines.append(f"| {s['condition']} | {label} | {s['n']} | {s['correct']} | {s['unsafe']} | "
                         f"{s['false_alarm']} | {s['wrong_knob']} | {s['abstain']} |")
    b, h = summarise(rows, "baseline"), summarise(rows, "holdpoint")
    pct = lambda x: "n/a" if x is None else f"{100 * x:.0f}%"
    headline = (f"Across {h['n']} checks, plain Muse Spark was confidently wrong {pct(b['confidently_wrong_rate'])} "
                f"of the time; Hold Point {pct(h['confidently_wrong_rate'])}, abstaining {pct(h['abstain_rate'])} "
                f"with a specific camera fix. When a knob was on, plain Spark said 'all off' "
                f"{pct(b['unsafe_rate_when_on'])} of the time; Hold Point {pct(h['unsafe_rate_when_on'])}.")
    text = "\n".join(lines) + "\n\n" + headline + "\n"
    (OUT / "summary.md").write_text(text)
    print("\n" + text)
    print(client.ledger.summary())


if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    main()
