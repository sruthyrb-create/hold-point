# Hold Point

**The first assistant for blind users that knows when it can't see.**

Hold Point is a voice-first assistant for blind and low-vision people that confirms the state of things at home ("are all the burners off?"). It answers only when two independent Meta models agree. Otherwise it says exactly what it can't see and how to move the camera.

> Built for the Meta Global AI Developer Hackathon (26 Oct - 4 Nov 2026). All code in this repo was written during the event.

## Status

Work in progress. Done so far:

- [x] Deterministic decision layer: answers only when both witnesses agree over several frames, otherwise abstains with a reason code
- [x] Knob-angle geometry (Witness B maths) from segmentation masks
- [x] Frame-quality checks (dark, blur, glare, too far) that veto evidence independently of either model
- [x] Short, action-first spoken answers with camera guidance
- [x] Defect-injection script that turns one clean walkthrough into labelled broken clips
- [x] Metrics: confident-wrong rate, abstention rate, correct rate, answer length, latency
- [x] Spend guard (hard cap, blocks unpriced models) and response cache (repeat calls are free)
- [x] Witness A: Muse Spark 1.3 per-frame structured readings (tested against a fake API; first real call pending)
- [ ] Witness B: SAM 3.1 tracking → masks
- [ ] Voice in (Muse Voice Transcribe), speech out (OS text-to-speech)
- [ ] Leaving-home agent (Muse Wonder), watch mode, home profile
- [ ] Local mode (Muse Glimmer + self-hosted SAM 3.1)

## How it works

```
camera frames ──┬─► Witness A: Muse Spark 1.3 (structured JSON per item) ─┐
                ├─► Witness B: SAM 3.1 masks → knob angle vs. OFF mark ───┼─► decision layer ─► short spoken answer
                └─► frame-quality checks (dark / blur / glare / far) ─────┘   (no AI; agree or abstain)
```

## Setup

```bash
conda create -n holdpoint python=3.12 -y
conda activate holdpoint
pip install -e ".[dev]"
pytest
```

API mode needs no GPU. Set `MODEL_API_KEY`, fill in `prices.json`, and set a cap with
`HOLDPOINT_BUDGET_USD` (default 120). Every call is logged to `.holdpoint/ledger.jsonl`.

## Build the test set

```bash
python eval/inject_defects.py my_stove.mp4 my_stove.truth.json clips/
python eval/metrics.py clips/ predictions.jsonl
```

See the docstring in `eval/inject_defects.py` for the ground-truth format.

## Limitations

- Not a safety-certified device. It assists, and does not replace, the user's own checks.
- Performance drops in bad lighting (by design it abstains rather than guesses).
