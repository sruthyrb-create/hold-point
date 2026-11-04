"""The user's home: fixed landmarks (the "imaginary points") and enrolled items.

Saved in HOME_DIR/home.json (git-ignored; HOME_DIR = $LASTSEEN_HOME or ./home). Example:

{
  "landmarks": [
    {"id": "stove", "name": "stove", "room": "kitchen", "sam_phrase": "stove"},
    {"id": "sofa",  "name": "sofa",  "room": "living room", "sam_phrase": "sofa"}
  ],
  "items": [
    {"id": "glasses", "name": "glasses", "aliases": ["specs"], "sam_phrase": "eyeglasses",
     "description": "black rectangular reading glasses with a thin metal bridge",
     "references": ["home/items/glasses_0.jpg"]}
  ]
}
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# All Last Seen data lives here (git-ignored). Inside Hold Point the default is home/lastseen.
HOME_DIR = Path(os.environ.get("LASTSEEN_HOME", "home/lastseen"))
HOME_FILE = HOME_DIR / "home.json"
MAX_ITEMS = 3


@dataclass
class Landmark:
    id: str
    name: str
    room: str
    sam_phrase: str


@dataclass
class Item:
    id: str
    name: str
    sam_phrase: str
    description: str
    aliases: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)   # enrolment crops (paths)

    @property
    def spoken_names(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Home:
    landmarks: list[Landmark] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    def item(self, item_id: str) -> Item:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)

    def landmark(self, landmark_id: str) -> Landmark | None:
        return next((l for l in self.landmarks if l.id == landmark_id), None)

    def item_names(self) -> dict[str, list[str]]:
        return {it.id: it.spoken_names for it in self.items}

    def add_item(self, item: Item) -> None:
        self.items = [i for i in self.items if i.id != item.id]
        if len(self.items) >= MAX_ITEMS:
            raise ValueError(f"At most {MAX_ITEMS} items. Remove one first.")
        self.items.append(item)

    def add_landmark(self, lm: Landmark) -> None:
        self.landmarks = [l for l in self.landmarks if l.id != lm.id] + [lm]

    def save(self, path: Path = HOME_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = HOME_FILE) -> "Home":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls([Landmark(**l) for l in raw.get("landmarks", [])],
                   [Item(**i) for i in raw.get("items", [])])
