"""Run the hide-and-seek evaluation: plain Muse Spark vs Last Seen.

    python scripts/lastseen_eval.py eval/truth.json --max-frames 12

For each trial: Last Seen watches the trial's frames with a FRESH memory, then
answers "where is it?". The baseline sends the same frames (up to 8, evenly
spaced) to plain Muse Spark and asks for the landmark. Costs real money:
roughly (items + 2) SAM images + 1 Spark call per processed frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lastseen.eval_metrics import Outcome, summarise   # noqa: E402
from lastseen.answer import answer_where      # noqa: E402
from lastseen.home import Home                # noqa: E402
from lastseen.memory import MemoryStore       # noqa: E402
from lastseen.meta_client import MetaClient, encode_frame   # noqa: E402
from lastseen.watch import Pipeline           # noqa: E402

BASELINE_SCHEMA = {"name": "Where", "schema": {"type": "object", "additionalProperties": False,
                   "required": ["landmark"], "properties": {"landmark": {"type": "string"}}}}


def frames_of(folder: Path, limit: int):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(paths) > limit:
        step = len(paths) / limit
        paths = [paths[int(i * step)] for i in range(limit)]
    return [(p, cv2.imread(str(p))) for p in paths]


def run_last_seen(client, home, trial, frames) -> Outcome:
    with tempfile.TemporaryDirectory() as d:
        store = MemoryStore(Path(d) / "m.jsonl")
        p = Pipeline(client, home, store, keep_snapshots=False)
        for k, (_, img) in enumerate(frames):
            p.process(img, now=1000.0 + 10 * k)
        ans = answer_where(trial["item"], store.for_item(trial["item"]), home, now=1000.0 + 10 * len(frames))
        said = []
        if ans["status"] == "two_places":
            said = sorted({s.landmark for s in store.for_item(trial["item"]) if s.status == "confirmed"})
        elif ans.get("sighting") is not None and ans["status"] in ("confirmed", "stale"):
            said = [ans["sighting"].landmark]
        return Outcome(trial["trial"], said, ans["status"], trial["landmark"])


def run_baseline(client, home, trial, frames) -> Outcome:
    imgs = [encode_frame(img) for _, img in frames[:8]]
    item = home.item(trial["item"])
    ids = [l.id for l in home.landmarks]
    raw = client.chat_json("muse-spark-1.3", "You answer questions about a home video.",
                           f"Where are my {item.name} ({item.description})? Answer with one of {ids}, "
                           "or 'unknown'.", imgs, BASELINE_SCHEMA, tag="eval_baseline")
    lm = raw.get("landmark", "unknown")
    return Outcome(trial["trial"], [lm] if lm in ids else [], "confirmed" if lm in ids else "never_seen",
                   trial["landmark"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args()
    trials = json.loads(Path(a.truth).read_text())
    home, client = Home.load(), MetaClient()
    ours, base = [], []
    for t in trials:
        frames = frames_of(Path(t["frames"]), a.max_frames)
        ours.append(run_last_seen(client, home, t, frames))
        if not a.skip_baseline:
            base.append(run_baseline(client, home, t, frames))
        print(t["trial"], ours[-1], base[-1] if base else "", flush=True)
    report = {"last_seen": summarise(ours), "plain_spark": summarise(base), "spent": client.ledger.summary()}
    print(json.dumps(report, indent=2))
    Path("lastseen_eval_results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
