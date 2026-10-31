"""Home profile: the user's own appliances, saved once at setup.

Lives in ./home/ (git-ignored: it holds images of the user's home).
  home/profile.json      names and layout, safe to edit by hand
  home/stove_refs.npz    each knob's OFF reference crop
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def save_stove(knobs: list[dict], refs: dict[str, np.ndarray], prompt: str, home: Path = HOME_DIR) -> None:
    home.mkdir(parents=True, exist_ok=True)
    profile_path = home / "profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    profile["stove"] = {"sam_prompt": prompt, "knobs": knobs}
    profile_path.write_text(json.dumps(profile, indent=2))
    np.savez_compressed(home / "stove_refs.npz", **refs)


def load_stove(home: Path = HOME_DIR) -> tuple[dict, dict[str, np.ndarray]]:
    profile_path = home / "profile.json"
    if not profile_path.exists():
        raise SystemExit("No home profile yet. Run: python scripts/setup_stove.py <all-off photo>")
    stove = json.loads(profile_path.read_text())["stove"]
    data = np.load(home / "stove_refs.npz")
    refs = {k["id"]: data[k["id"]] for k in stove["knobs"]}   # keeps left-to-right order
    return stove, refs
