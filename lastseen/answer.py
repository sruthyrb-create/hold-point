"""Turn sightings into one short spoken answer. Plain code, no AI.

Honesty rules:
- always say how long ago;
- never seen -> say so, no guess;
- the newest sighting's place wasn't confirmed by both witnesses -> say "possibly"
  and give the last confirmed place too;
- two confirmed places close together in time -> say both;
- old sightings -> say it may have moved.
"""
from __future__ import annotations

import time

from .home import Home
from .memory import CONFIRMED, Sighting
from .spatial import place_phrase

STALE_S = 30 * 60        # older than this -> "may have moved"
AMBIGUITY_S = 120        # two different confirmed places within this window -> say both


def age_phrase(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    minutes = round(seconds / 60)
    if minutes < 90:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = round(seconds / 3600)
    return f"about {hours} hour{'s' if hours != 1 else ''} ago"


def _plural(name: str) -> bool:
    return name.endswith("s")


def _place(s: Sighting, home: Home) -> str | None:
    lm = home.landmark(s.landmark) if s.landmark else None
    if lm is None or s.relation is None:
        return None
    return place_phrase(s.relation, lm.name, lm.room)


def _key(s: Sighting) -> tuple:
    return (s.landmark, s.relation)


def answer_where(item_id: str, sightings: list[Sighting], home: Home, now: float | None = None) -> dict:
    """Returns {"text", "status", "sighting"}; status is one of
    never_seen, confirmed, two_places, unsure, stale."""
    now = now if now is not None else time.time()
    item = home.item(item_id)
    name = item.name
    they = "they" if _plural(name) else "it"
    were = "were" if _plural(name) else "was"
    mine = [s for s in sightings if s.item == item_id]
    if not mine:
        return {"text": f"I haven't seen your {name} yet, so I won't guess.",
                "status": "never_seen", "sighting": None}

    latest = mine[-1]
    confirmed = [s for s in mine if s.status == CONFIRMED and _place(s, home)]
    last_ok = confirmed[-1] if confirmed else None

    # newest sighting has no confirmed place
    # (an unconfirmed newer sighting at the SAME claimed place still supports last_ok)
    if last_ok is None or (latest is not last_ok and _key(latest) != _key(last_ok)):
        claim = _place(latest, home)
        if claim:
            text = (f"Not sure. I saw your {name} {age_phrase(now - latest.t)}, possibly {claim}, "
                    f"but couldn't confirm the place.")
        elif latest.seen_on:
            text = (f"Not sure of the exact place. I saw your {name} {age_phrase(now - latest.t)} "
                    f"on what looked like a {latest.seen_on}, but no landmark I know was in view.")
        else:
            text = (f"Not sure. I saw your {name} {age_phrase(now - latest.t)}, "
                    f"but no landmark I know was in view.")
        if last_ok is not None:
            text += f" Before that, {they} {were} {_place(last_ok, home)}, {age_phrase(now - last_ok.t)}."
        return {"text": text, "status": "unsure", "sighting": latest}

    # two different confirmed places close together in time
    rivals = [s for s in confirmed if _key(s) != _key(last_ok) and last_ok.t - s.t <= AMBIGUITY_S]
    if rivals:
        other = rivals[-1]
        text = (f"Two places. Most recently {_place(last_ok, home)}, {age_phrase(now - last_ok.t)}. "
                f"Also {_place(other, home)}, {age_phrase(now - other.t)}.")
        return {"text": text, "status": "two_places", "sighting": last_ok}

    text = f"Your {name}: {_place(last_ok, home)}, {age_phrase(now - last_ok.t)}."
    if now - last_ok.t > STALE_S:
        text += f" That was a while ago; {they} may have moved."
        return {"text": text, "status": "stale", "sighting": last_ok}
    return {"text": text, "status": "confirmed", "sighting": last_ok}
