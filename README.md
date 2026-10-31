# Hold Point

**The leaving-home check that won't guess.**

Say *"I'm leaving, check the house,"* and Hold Point walks you through the stove, oven, taps and doors. It verifies each one with two independent Meta models and says "all clear" only when it can prove it. Anything it can't confirm, it names, and it tells you exactly how to move the camera.

It's built first for people who can't double-check an AI's answer by looking: **blind and low-vision people**. It works just as well for **older adults** and anyone who has ever turned back halfway down the street to check the stove.

> Built for the Meta Global AI Developer Hackathon (26 Oct - 4 Nov 2026), Push the Frontier track. All code in this repo was written during the event.

---

## The problem

Leaving the house means trusting that everything is off. For someone who can't see the knobs, that trust has nowhere to come from. Today's AI assistants fill the gap by **guessing confidently**.

- **Cooking is the leading cause of home fires and home-fire injuries in the US.** In 2021 there were an estimated **170,000 home cooking fires**, causing 135 deaths, 3,000 injuries and over $494 million in losses. **Unattended equipment** was the leading factor in non-confined cooking fires (37%). ([US Fire Administration](https://www.usfa.fema.gov/prevention/home-fires/prevent-fires/cooking/))
- **Older adults face much higher stakes.** In 2023, people 65 and over were **2.5 times** more likely to die in a fire than the general population, and those 85 and over **3.4 times**. ([US Fire Administration](https://usfa.fema.gov/statistics/deaths-injuries/older-adults.html))
- **AI assistants answer confidently when they shouldn't.** A study built with 106 blind and visually impaired people tested 13 AI models and found they "often gave confident answers to the hardest questions, even when the input was insufficient." The researchers call for abstention to be treated as being as important as correct answers. ([University of Copenhagen & Heriot-Watt, ACL 2025](https://sciencenews.dk/en/when-ai-sounds-certain-blind-users-have-to-trust-it-even-when-it-is-guessing))
- **Answers are too long and don't say what to do next.** In 418 interactions with Be My AI, answers were over 10 times longer than the questions, with little guidance on positioning the camera. ([University of Maryland, 2026](https://www.digitaltrends.com/cool-tech/ai-apps-blind-users-umd-study/))
- **Appliances are getting harder, not easier.** 52% of 627 people surveyed by Vision Australia said newer appliances are less accessible than older ones, mostly because of touchscreens and flat panels. ([CHOICE](https://www.choice.com.au/home-and-living/kitchen/articles/modern-appliances-are-leaving-the-blind-behind))

**A confident wrong "all burners off" is worse than no answer.** Hold Point is built so that it can't give one.

---

## What's new

Vision assistants describe what they see. Hold Point is built to **know what it can't see**.

1. **Two witnesses must agree.** Muse Spark reads each item; SAM 3.1 measures it geometrically. If they disagree, Hold Point doesn't answer.
2. **Knob positions are measured, not guessed.** SAM 3.1 finds the knob and its pointer, and Hold Point computes the angle against that specific stove's OFF mark.
3. **Every "I can't tell" becomes an instruction.** Each reason for abstaining (out of frame, covered, too dark, blurry, too far, glare) maps to a specific spoken fix, then it checks again.
4. **Short, action-first answers.** "Three off. Can't confirm back-left. Move the phone left." Danger first, never a scene description.
5. **An agent that finishes the job.** It plans the checklist from your home profile, guides you item by item, rechecks anything it abstained on, and ends with one sentence.
6. **A record you can come back to.** "Checked 9:02 PM: all off, back door locked." That answers the *"did I leave it on?"* worry after you've left.
7. **Measured honesty.** On a labelled test set built by injecting defects into real footage, we report how often Hold Point is **confidently wrong** compared with a plain AI asked the same question.

> **Headline result:** *a plain AI was confidently wrong __% of the time; Hold Point __%, abstaining __% of the time with a specific fix.* (filled in after evaluation)

---

## Beyond the front door

The leaving-home check is the first application of a general capability: **visual verification that knows its own limits**. The same engine (two independent witnesses, a deterministic decision layer and reason-coded guidance) applies wherever a wrong confident answer about a physical state has real consequences:

- **Night-time lock-up:** doors locked, windows shut, oven off before bed.
- **Reading displays nobody designed to be accessible:** oven temperature, thermostat, microwave timer, washer cycle ("It shows 375, not 350").
- **Watch mode:** "Tell me when the oven reaches 350" or "when the washer finishes."
- **Peace of mind for families:** with the user's consent, a verified check record shared with a family member, rather than a phone call asking "did you turn it off?"
- **Closing checklists for small businesses:** fryers, coffee machines and back doors at the end of a shift.
- **A safety layer for other AI agents.** Any assistant that acts on what a camera sees, on smart glasses or in a robot, needs to know when it can't see. Hold Point's abstention layer is a standalone component designed to be reused.

The capture layer targets 720p at 30 fps, matching what Ray-Ban Meta glasses stream, so the same engine can move to glasses with no change to the logic.

---

## How it works

```
                        ┌─► Witness A: Muse Spark 1.3 (structured JSON per item, per frame) ─┐
camera frames ──────────┼─► Witness B: SAM 3.1 tracking → knob angle vs. OFF mark ───────────┼─► decision layer ─► short spoken answer
                        └─► frame-quality checks (dark / blur / glare / too far) ────────────┘   (no AI: agree or abstain)
                                                                                                        │
"I'm leaving, check the house" ─► Muse Wonder plans the checklist from the home profile ───────────────┘─► one-sentence summary + saved record
```

| Meta model | Role |
|---|---|
| Muse Spark 1.3 | Witness A: per-frame structured readings of each knob, display and indicator |
| SAM 3.1 | Witness B: segments and tracks knobs, pointers and displays across frames |
| Muse Wonder | Leaving-home agent: plans the checklist, sequences checks, writes the final summary |
| Muse Voice Transcribe | Voice input |
| Muse Glimmer (open weights) | Local mode: the vision witness running on self-hosted hardware, so home video never leaves your control |

---

## Status

- [x] Deterministic decision layer: answers only when both witnesses agree over several frames, otherwise abstains with a reason code
- [x] Knob-angle geometry (Witness B maths) from segmentation masks
- [x] Frame-quality checks that veto evidence independently of either model
- [x] Short, action-first spoken answers with camera guidance
- [x] Defect-injection script that turns one clean walkthrough into labelled broken clips
- [x] Metrics: confident-wrong rate, abstention rate, correct rate, answer length, latency
- [x] Spend guard (hard cap, blocks unpriced models) and response cache (repeat calls are free)
- [x] Witness A: Muse Spark 1.3 per-frame structured readings (tested against a fake API; first real call pending)
- [ ] Witness B: SAM 3.1 tracking → masks
- [ ] Leaving-home agent (Muse Wonder) and home profile
- [ ] Voice in (Muse Voice Transcribe), speech out (OS text-to-speech)
- [ ] Saved check record
- [ ] Watch mode
- [ ] Local mode (Muse Glimmer + self-hosted SAM 3.1)

---

## Setup

```bash
conda create -n holdpoint python=3.12 -y
conda activate holdpoint
pip install -e ".[dev]"
pytest
```

API mode needs no GPU. Set `MODEL_API_KEY`, and optionally set a spending cap with `HOLDPOINT_BUDGET_USD` (default $120). Prices are in `prices.json`; every call is logged to `.holdpoint/ledger.jsonl`, and repeated requests are served from a local cache at no cost.

### Build the test set

```bash
python eval/inject_defects.py my_stove.mp4 my_stove.truth.json clips/
python eval/metrics.py clips/ predictions.jsonl
```

See the docstring in `eval/inject_defects.py` for the ground-truth format.

---

## Limitations

- **Not a safety-certified device.** It assists, and does not replace, the user's own checks or a smoke alarm.
- **It abstains more in poor conditions.** In bad lighting or with a shaky camera it will say what it can't see rather than guess. That is the design, but it means more prompts to adjust the camera.
- **It verifies what it's set up to check.** Items need to be added to the home profile once, ideally with a sighted helper.

## Credits

Speech output uses the device's built-in text-to-speech. All model inference uses Meta models.
