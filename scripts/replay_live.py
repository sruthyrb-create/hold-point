"""Re-run the decision on saved live checks, without calling any model (free).

    python scripts/replay_live.py home/live
    python scripts/replay_live.py home/live --off 6      # compare an OFF window for the frame middle

Every live check saves both witnesses' per-frame readings (home/live/<time>/result.json).
This rebuilds them and re-decides, so a threshold change can be checked against real
checks before trying it at the stove.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint import check as check_mod  # noqa: E402
from holdpoint.decision import DecisionConfig, decide  # noqa: E402
from holdpoint.guidance import all_off_answer  # noqa: E402
from holdpoint.schemas import FrameReadings, ItemReading, ReasonCode  # noqa: E402


def rebuild(result: dict):
    n = result["frames"]
    vlm = {i: [] for i in range(n)}
    geo = {i: [] for i in range(n)}
    for it in result["items"]:
        for x in it["spark"]:
            vlm[x["frame"] - 1].append(ItemReading(
                item_id=it["id"], visible=x["state"] is not None, state=x["state"],
                confidence=x["confidence"], reason_code=ReasonCode(x["reason"])))
        for x in it["sam"]:
            measured = x["turn_deg"] is not None and x["reason"] in ("OK", "LOW_EVIDENCE")
            geo[x["frame"] - 1].append(ItemReading(
                item_id=it["id"], visible=measured, state=x["state"],
                confidence=1.0 if x["state"] else 0.0, reason_code=ReasonCode(x["reason"]),
                angle_deg=x["turn_deg"]))
    vlm_f = [FrameReadings(frame_index=i, witness="vlm", items=v) for i, v in vlm.items() if v]
    geo_f = [FrameReadings(frame_index=i, witness="geometry", items=v) for i, v in geo.items()]
    quality = {int(k) - 1: [ReasonCode(q) for q in v] for k, v in result.get("quality", {}).items()}
    return vlm_f, geo_f, quality


def replay(result: dict, off_deg: float) -> str:
    if result.get("errors"):
        return result["answer"]
    vlm, geo, quality = rebuild(result)
    ids = [it["id"] for it in result["items"]]
    names = {it["id"]: it["name"] for it in result["items"]}
    saved = check_mod.CONSENSUS_OFF_DEG
    check_mod.CONSENSUS_OFF_DEG = off_deg
    try:
        judged = check_mod.geometry_consensus(geo, ids)
    finally:
        check_mod.CONSENSUS_OFF_DEG = saved
    verdicts = decide(ids, vlm, judged, quality=quality, cfg=DecisionConfig(min_visible_frames=3))
    return all_off_answer(verdicts, names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--off", type=float, default=check_mod.CONSENSUS_OFF_DEG,
                    help="OFF window (deg) for the middle of agreeing frames")
    ap.add_argument("--base", type=float, default=6.0, help="window to compare against")
    a = ap.parse_args()
    for d in sorted(Path(a.folder).iterdir()):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        old, new = replay(r, a.base), replay(r, a.off)
        mark = "   " if old == new else " * "
        print(f"{d.name[:15]}{mark}+/-{a.base:g}: {old:55} +/-{a.off:g}: {new}")


if __name__ == "__main__":
    main()
