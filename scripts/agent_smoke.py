"""Try the leaving-home agent with the REAL Muse Wonder but PRETEND check results
(no camera, no stove). Confirms Wonder's tool calling works. Cost: about 2 cents.

    python scripts/agent_smoke.py

Script: the first stove check can't confirm a knob ("Move closer."), the second
finds all four off. Expected: Wonder re-checks once, then finishes; "planner: wonder".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holdpoint import agent  # noqa: E402
from holdpoint.meta_client import MetaClient  # noqa: E402

PRETEND = [
    {"answer": "Three off. Can't confirm knob 1. Move closer.", "all_clear": False, "danger": False,
     "guidance": "Move closer."},
    {"answer": "Yes, all four off.", "all_clear": True, "danger": False, "guidance": None},
]

if __name__ == "__main__":
    if "MODEL_API_KEY" not in os.environ:
        raise SystemExit("Set MODEL_API_KEY first.")
    client = MetaClient()
    s, step = agent.start(client, ["stove", "back door"], {"stove"}, "I'm leaving, check the house")
    print(f"-> {step.action} {step.item or ''}: {step.say}")
    results = iter(PRETEND)
    while step.action == "check":
        result = next(results, PRETEND[-1])
        print(f"   (pretend check result: {result['answer']})")
        step = agent.submit(client, s, result)
        print(f"-> {step.action} {step.item or ''}: {step.say}")
    print("\nplanner:", s.planner)
    for t in s.trace:
        print(f"  {t['model']:16} {t['call']:22} {t['said']}")
    print(client.ledger.summary())
