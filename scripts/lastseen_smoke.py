"""First live run with photos, no phone needed. Costs a few cents (hard cap $0.50).

Take these JPG photos (phone, normal room light) and put them in one folder:
  item1.jpg item2.jpg item3.jpg   the item (e.g. glasses) on a plain surface, 3 different sides
  landmark.jpg                    the landmark (e.g. the stove), whole thing in view
  scene.jpg                       the item placed next to the landmark, both in view
  empty.jpg                       (optional) the same scene WITHOUT the item

Then:
  python scripts/lastseen_smoke.py PHOTO_FOLDER --item glasses --phrase eyeglasses --landmark stove --room kitchen

It enrols the item, adds the landmark, watches scene.jpg (and empty.jpg), and asks
"where are my glasses?". Writes home/lastseen/smoke_scene.jpg with SAM's boxes drawn on it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lastseen.answer import answer_where          # noqa: E402
from lastseen.budget import Ledger                # noqa: E402
from lastseen.enrol import add_landmark, enrol_item   # noqa: E402
from lastseen.home import Home                    # noqa: E402
from lastseen.memory import MemoryStore           # noqa: E402
from lastseen.meta_client import MetaClient, prepare_frame   # noqa: E402
from lastseen.watch import Pipeline               # noqa: E402


def img(folder: Path, name: str, required=True):
    p = folder / name
    if not p.exists():
        if required:
            sys.exit(f"Missing {p}")
        return None
    im = cv2.imread(str(p))
    if im is None:
        sys.exit(f"Can't read {p} (HEIC? save it as JPG)")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--item", required=True)
    ap.add_argument("--phrase", default="")
    ap.add_argument("--landmark", required=True)
    ap.add_argument("--room", required=True)
    a = ap.parse_args()
    folder = Path(a.folder)

    ledger = Ledger()
    ledger.cap = ledger.spent() + 0.50          # this script can spend at most 50 cents
    client = MetaClient(ledger)
    home = Home.load()

    print("1. Enrolling the item (3 SAM calls + 1 Spark call)...")
    item = enrol_item(client, home, [img(folder, f"item{i}.jpg") for i in (1, 2, 3)], a.item, a.phrase)
    print("   ", item.description, "| crops:", item.references)

    print("2. Adding the landmark (1 SAM call)...")
    lm = add_landmark(client, home, img(folder, "landmark.jpg"), a.landmark, a.room)
    print("   ", lm)

    store = MemoryStore()
    p = Pipeline(client, Home.load(), store)
    print("3. Watching scene.jpg...")
    scene = img(folder, "scene.jpg")
    r = p.process(scene)
    print(json.dumps({k: r[k] for k in ("skipped", "candidates", "sightings", "calls")}, indent=1, default=str))
    out = prepare_frame(scene).copy()
    for c in r["candidates"]:
        x0, y0, x1, y1 = c["box"]
        ok = c["spark"] and c["spark"]["is_users_item"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 0) if ok else (0, 0, 255), 3)
    for l in r.get("landmarks", []):                 # SAM's landmark boxes in blue
        x0, y0, x1, y1 = l["box"]
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 120, 0), 3)
        cv2.putText(out, f'{l["id"]}: {l["relation"]}', (x0 + 5, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 0), 2)
    print("   landmark boxes:", r.get("landmarks"))
    cv2.imwrite("home/lastseen/smoke_scene.jpg", out)

    empty = img(folder, "empty.jpg", required=False)
    if empty is not None:
        print("4. Watching empty.jpg (should store nothing new)...")
        p.state.last_t = 0                    # don't skip it as "unchanged"
        r2 = p.process(empty)
        print("   sightings:", r2["sightings"], "| skipped:", r2["skipped"])

    print("5. Asking: where are my", item.name, "?")
    print("   ->", answer_where(item.id, store.for_item(item.id), Home.load())["text"])
    print("Spent:", ledger.summary(), "| overlay: home/lastseen/smoke_scene.jpg")


if __name__ == "__main__":
    main()
