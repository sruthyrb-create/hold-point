"""The watching loop, one frame at a time.

frame -> free local checks (too dark / blurry / nothing changed -> skip, $0)
      -> Witness A: SAM 3.1 looks for each enrolled item by name (one call per item)
      -> Witness B: Muse Spark confirms each candidate is the user's item and names
         the nearest landmark from the user's list (one call for all candidates)
      -> if Spark named a landmark: SAM 3.1 finds that landmark, and plain geometry
         checks Spark's relation ("left of the stove") against the two boxes
      -> store a sighting: CONFIRMED (item and place agreed) or PLACE_UNVERIFIED.

Nothing is stored when the witnesses disagree that it's the user's item, or when
two candidates look equally like the same item (we can't tell which is theirs).
Not seeing an item is not stored: absence in one view says nothing about where it is.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import sam as sam_mod
from . import witness_spark
from .home import HOME_DIR, Home
from .memory import CONFIRMED, PLACE_UNVERIFIED, MemoryStore, Sighting
from .meta_client import prepare_frame
from .spatial import direction_in_frame, relation, relations_agree

DARK_MEAN = 45
BLUR_VAR = 20
CHANGE_MIN = 6.0          # mean abs grey difference (0-255) on a 64x48 thumbnail
CHANGE_MAX_AGE_S = 60     # process at least once a minute even if nothing changed
MIN_SAM_CONF = 0.3
TIE_MARGIN = 0.15         # two candidates for one item closer than this -> abstain
SNAPSHOT_SIDE = 160
SNAPSHOT_DIR = HOME_DIR / "snapshots"


def frame_problem(frame: np.ndarray) -> str | None:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if grey.mean() < DARK_MEAN:
        return "too_dark"
    if cv2.Laplacian(grey, cv2.CV_64F).var() < BLUR_VAR:
        return "blur"
    return None


def thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


@dataclass
class WatchState:
    last_thumb: np.ndarray | None = None
    last_t: float = 0.0
    frames_seen: int = 0
    frames_processed: int = 0

    def changed(self, frame: np.ndarray, now: float) -> bool:
        t = thumb(frame)
        if self.last_thumb is None or now - self.last_t >= CHANGE_MAX_AGE_S:
            return True
        return float(np.abs(t - self.last_thumb).mean()) >= CHANGE_MIN

    def mark(self, frame: np.ndarray, now: float) -> None:
        self.last_thumb, self.last_t = thumb(frame), now
        self.frames_processed += 1


@dataclass
class Pipeline:
    """The API calls are injectable so tests (and the eval) can use fakes."""
    client: object
    home: Home
    store: MemoryStore
    segment: object = None       # (client, image, phrase, tag) -> list[SamObject]
    confirm: object = None       # (client, frame, candidates, home, tag) -> list[Match]
    keep_snapshots: bool = True
    state: WatchState = field(default_factory=WatchState)

    def __post_init__(self):
        self.segment = self.segment or (lambda c, img, p, tag="": sam_mod.segment_image(c, img, p, tag=tag))
        self.confirm = self.confirm or witness_spark.confirm

    # ------------------------------------------------------------ helpers
    def _timed(self, calls: list, name: str, fn, *args):
        t0 = time.time()
        out = fn(*args)
        calls.append({"model": name, "seconds": round(time.time() - t0, 2)})
        return out

    def _find_items(self, frame, calls, item_ids) -> list[witness_spark.Candidate]:
        def one(iid):
            it = self.home.item(iid)
            return iid, self._timed(calls, f"sam-3.1 '{it.sam_phrase}'", self.segment,
                                    self.client, frame, it.sam_phrase, "watch")
        with ThreadPoolExecutor(max_workers=3) as ex:
            found = list(ex.map(one, item_ids))
        cands = []
        for iid, objs in found:
            for o in objs:
                if o.confidence is None or o.confidence >= MIN_SAM_CONF:
                    cands.append(witness_spark.Candidate(iid, o))
        return cands

    def _snapshot(self, frame, box, frame_id, item) -> str | None:
        if not self.keep_snapshots:
            return None
        crop = witness_spark.crop_context(frame, box, factor=1.6, max_side=SNAPSHOT_SIDE)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOT_DIR / f"{frame_id}_{item}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    # ------------------------------------------------------------ main step
    def process(self, frame_bgr: np.ndarray, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        frame = prepare_frame(frame_bgr)
        self.state.frames_seen += 1
        result = {"processed": False, "skipped": None, "candidates": [], "sightings": [], "calls": []}

        problem = frame_problem(frame)
        if problem:
            result["skipped"] = problem
            return result
        if not self.state.changed(frame, now):
            result["skipped"] = "unchanged"
            return result
        self.state.mark(frame, now)
        result["processed"] = True
        calls = result["calls"]
        frame_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + uuid.uuid4().hex[:4]

        cands = self._find_items(frame, calls, [it.id for it in self.home.items])
        if not cands:
            return result
        matches = self._timed(calls, "muse-spark-1.3", self.confirm, self.client, frame, cands, self.home, "watch")
        by_idx = {m.candidate: m for m in matches}
        for i, c in enumerate(cands):
            m = by_idx.get(i)
            result["candidates"].append({
                "item": c.item, "box": list(c.obj.box), "sam_conf": c.obj.confidence,
                "spark": None if m is None else {"is_users_item": m.is_users_item, "confidence": m.confidence,
                                                 "landmark": m.landmark, "relation": m.relation,
                                                 "reason": m.reason, "surface": m.surface}})

        # best accepted candidate per item; abstain on near-ties
        best: dict[str, tuple[witness_spark.Candidate, witness_spark.Match]] = {}
        tied = set()
        for i, c in enumerate(cands):
            m = by_idx.get(i)
            if m is None or not m.is_users_item or m.confidence < witness_spark.MIN_MATCH:
                continue
            if c.item in best:
                other = best[c.item][1].confidence
                if abs(other - m.confidence) < TIE_MARGIN:
                    tied.add(c.item)
                if m.confidence <= other:
                    continue
            best[c.item] = (c, m)
        result["abstained"] = sorted(tied)

        lm_cache: dict[str, list] = {}
        for iid, (c, m) in best.items():
            if iid in tied:
                continue
            status, lm_id, rel, geo = PLACE_UNVERIFIED, m.landmark, m.relation, None
            lm = self.home.landmark(m.landmark) if m.landmark else None
            if lm is not None:
                if lm.id not in lm_cache:
                    lm_cache[lm.id] = self._timed(calls, f"sam-3.1 '{lm.sam_phrase}'", self.segment,
                                                  self.client, frame, lm.sam_phrase, "landmark")
                rels = [relation(c.obj.box, o.box, o.mask) for o in lm_cache[lm.id]]
                result.setdefault("landmarks", []).extend(
                    {"id": lm.id, "box": list(o.box), "relation": r} for o, r in zip(lm_cache[lm.id], rels))
                rels = [r for r in rels if r is not None]
                agreeing = [r for r in rels if relations_agree(m.relation, r)]
                geo = rels
                if agreeing:
                    status = CONFIRMED
                    # keep the more specific of the two descriptions
                    rel = m.relation if m.relation != "near" else agreeing[0]
            s = Sighting(item=iid, t=now, status=status, landmark=lm_id, relation=rel,
                         room=lm.room if lm else None, spark_conf=m.confidence, sam_conf=c.obj.confidence,
                         snapshot=self._snapshot(frame, c.obj.box, frame_id, iid), frame_id=frame_id,
                         seen_on=m.surface)
            self.store.add(s)
            result["sightings"].append({**s.__dict__, "geometry_relations": geo})
        return result

    # ------------------------------------------------------------ "found it" step
    def find(self, frame_bgr: np.ndarray, item_id: str, landmark_id: str | None = None) -> dict:
        """User is searching: is the item in THIS view, and which way? If not, and a
        landmark is given, also report whether that landmark is in view (so the guide
        knows whether the user has reached the right spot)."""
        frame = prepare_frame(frame_bgr)
        it = self.home.item(item_id)
        calls = []
        problem = frame_problem(frame)
        if problem:
            fix = "It's too dark here. Turn on a light." if problem == "too_dark" else "Hold the phone still."
            return {"found": False, "text": fix, "calls": calls, "quality": problem}
        cands = self._find_items(frame, calls, [item_id])
        matches = self._timed(calls, "muse-spark-1.3", self.confirm, self.client, frame, cands, self.home, "find") if cands else []
        ok = [(cands[m.candidate], m) for m in matches
              if m.is_users_item and m.confidence >= witness_spark.MIN_MATCH]
        name = it.name
        if not ok:
            out = {"found": False, "text": f"I can't see your {name} in this view. Turn slowly.", "calls": calls}
            lm = self.home.landmark(landmark_id) if landmark_id else None
            if lm is not None:
                objs = self._timed(calls, f"sam-3.1 '{lm.sam_phrase}'", self.segment,
                                   self.client, frame, lm.sam_phrase, "find_landmark")
                out["landmark_seen"] = bool(objs)
            return out
        if len(ok) > 1:
            return {"found": False, "text": f"I see two things that could be your {name}. Move closer to one.",
                    "calls": calls}
        c, _ = ok[0]
        h, w = frame.shape[:2]
        return {"found": True, "text": f"Found your {name}: {direction_in_frame(c.obj.box, w, h)}.",
                "box": list(c.obj.box), "calls": calls}
