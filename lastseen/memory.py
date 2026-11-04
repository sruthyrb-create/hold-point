"""Sighting memory: what was seen, where, when, and how sure.

One JSON line per accepted sighting in home/memories.jsonl (git-ignored).
Privacy: no full frames are kept; only a small crop of the item (optional),
and everything older than RETENTION_HOURS is deleted on every load.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .home import HOME_DIR

MEMORY_FILE = HOME_DIR / "memories.jsonl"
RETENTION_HOURS = float(os.environ.get("LASTSEEN_RETENTION_HOURS", "24"))
_LOCK = threading.Lock()

# status of a sighting
CONFIRMED = "confirmed"            # both witnesses agree it's the item AND agree on the place
PLACE_UNVERIFIED = "place_unverified"  # both agree it's the item; place not confirmed by both


@dataclass
class Sighting:
    item: str
    t: float
    status: str
    landmark: str | None = None     # landmark id
    relation: str | None = None     # spatial.RELATIONS
    room: str | None = None
    spark_conf: float = 0.0
    sam_conf: float | None = None
    snapshot: str | None = None     # path to a small item crop, if kept
    frame_id: str | None = None
    seen_on: str | None = None      # Spark's few-word description of the surface (unverified)


class MemoryStore:
    def __init__(self, path: Path = MEMORY_FILE, retention_hours: float = RETENTION_HOURS):
        self.path = path
        self.retention_s = retention_hours * 3600
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, s: Sighting) -> None:
        with _LOCK, self.path.open("a") as f:
            f.write(json.dumps(asdict(s)) + "\n")

    def all(self, now: float | None = None) -> list[Sighting]:
        """All sightings still inside the retention window, oldest first. Expired
        lines (and their snapshots) are deleted from disk."""
        now = now if now is not None else time.time()
        if not self.path.exists():
            return []
        keep, drop = [], []
        with _LOCK:
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    s = Sighting(**json.loads(line))
                except (TypeError, ValueError):
                    continue
                (keep if now - s.t <= self.retention_s else drop).append(s)
            if drop:
                self.path.write_text("".join(json.dumps(asdict(s)) + "\n" for s in keep))
                for s in drop:
                    if s.snapshot:
                        Path(s.snapshot).unlink(missing_ok=True)
        return sorted(keep, key=lambda s: s.t)

    def for_item(self, item: str, now: float | None = None) -> list[Sighting]:
        return [s for s in self.all(now) if s.item == item]

    def forget(self, item: str | None = None, now: float | None = None) -> int:
        """Delete all memories (or one item's). Returns how many were removed."""
        sightings = self.all(now)
        gone = [s for s in sightings if item is None or s.item == item]
        keep = [s for s in sightings if s not in gone]
        with _LOCK:
            self.path.write_text("".join(json.dumps(asdict(s)) + "\n" for s in keep))
        for s in gone:
            if s.snapshot:
                Path(s.snapshot).unlink(missing_ok=True)
        return len(gone)
