"""Home profile: the user's own appliances, saved once at setup.

Lives in ./home/ (git-ignored: it describes the user's home).
  home/profile.json      names, layout and each knob's OFF bar direction; safe to edit by hand
"""
from __future__ import annotations

import json
from pathlib import Path

HOME_DIR = Path(__file__).resolve().parent.parent / "home"
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth"]


def knob_entries(n: int) -> list[dict]:
    out = []
    for i in range(n):
        side = "left" if i < n / 2 else "right"
        out.append({
            "id": f"knob_{i + 1}",
            "name": f"knob {i + 1}",                       # spoken name; rename e.g. "back-left burner"
            "position": side,                               # used for "move the phone left/right"
            "description": f"the {ORDINAL[i]} control knob from the left, of {n} knobs in a row",
        })
    return out


def save_stove(knobs: list[dict], prompt: str, home: Path = HOME_DIR) -> None:
    """Each knob dict must include "off_direction" (degrees, from setup)."""
    home.mkdir(parents=True, exist_ok=True)
    profile_path = home / "profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    profile["stove"] = {"sam_prompt": prompt, "knobs": knobs}
    profile_path.write_text(json.dumps(profile, indent=2))


def load_stove(home: Path = HOME_DIR) -> tuple[dict, dict[str, float]]:
    profile_path = home / "profile.json"
    if not profile_path.exists():
        raise SystemExit("No home profile yet. Run: python scripts/setup_stove.py <all-off photo>")
    stove = json.loads(profile_path.read_text())["stove"]
    if any("off_direction" not in k for k in stove["knobs"]):
        raise SystemExit("Profile is from an older version. Re-run scripts/setup_stove.py.")
    refs = {k["id"]: float(k["off_direction"]) for k in stove["knobs"]}   # left-to-right order
    return stove, refs
