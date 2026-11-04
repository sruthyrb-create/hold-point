"""The "find my keys" step of Hold Point's leaving-home check.

A few phone frames -> for each frame, SAM 3.1 looks for the item and Muse Spark
confirms it's the user's own (both must agree, as in watching). Stops at the
first verified view. If not found, says where it was last seen (from memory)
and whether that landmark is in view, so the next attempt can be guided.

Returns a result shaped like Hold Point's other checks, plus "found" and
kind="find". "found" is the only thing that can make the agent say "found".
"""
from __future__ import annotations

import time

from .answer import answer_where
from .home import Home
from .memory import MemoryStore
from .spatial import place_phrase
from .watch import Pipeline


def last_known(item_id: str, home: Home, store: MemoryStore):
    """(landmark_id or None, spoken memory answer) for the item."""
    ans = answer_where(item_id, store.for_item(item_id), home)
    s = ans.get("sighting")
    return (s.landmark if s is not None else None), ans["text"]


def run_find_check(client, images, item_id: str, home: Home | None = None,
                   store: MemoryStore | None = None, pipeline: Pipeline | None = None) -> dict:
    home = home or Home.load()
    store = store or MemoryStore()
    it = home.item(item_id)
    p = pipeline or Pipeline(client, home, store, keep_snapshots=False)
    spent0 = client.ledger.spent() if hasattr(client, "ledger") else 0.0
    landmark_id, memory_text = last_known(item_id, home, store)
    lm = home.landmark(landmark_id) if landmark_id else None
    t0 = time.perf_counter()
    calls, look = [], None
    for k, img in enumerate(images):
        last = k == len(images) - 1
        look = p.find(img, item_id, landmark_id if last else None)   # landmark check once, on the last view
        for c in look.get("calls", []):
            model, _, phrase = c["model"].partition(" ")        # e.g. "sam-3.1 'keys'"
            job = f"finds {phrase}" if phrase else f"is it your {it.name}?"
            calls.append({"model": model, "job": f"view {k + 1}: {job}", "seconds": c["seconds"]})
        if look.get("found"):
            break
    found = bool(look and look.get("found"))
    if found:
        answer, guidance = look["text"], None
    else:
        answer = f"I can't see your {it.name} here."
        if look and look.get("quality"):
            guidance = look["text"]
        elif lm is not None and look and look.get("landmark_seen"):
            guidance = f"I can see the {lm.name}. Lower the phone a little and turn slowly."
        elif lm is not None:
            guidance = f"Last seen {place_phrase('near', lm.name, lm.room)}. Go there and turn slowly."
        else:
            guidance = "Turn slowly."
        answer = f"{answer} {guidance}"
    spent1 = client.ledger.spent() if hasattr(client, "ledger") else 0.0
    return {"kind": "find", "item": item_id, "found": found, "all_clear": False, "danger": False,
            "answer": answer, "guidance": guidance, "memory": memory_text, "calls": calls,
            "cost_usd": round(spent1 - spent0, 4), "seconds": round(time.perf_counter() - t0, 2)}
