# Hold Point

**The leaving-home check that won't guess.**

Say *"I'm leaving, check the house,"* and Hold Point walks you to the stove and checks every knob with two independent Meta models. It says "all clear" only when it can prove it. Anything it can't confirm, it names, and it tells you exactly how to move the camera. Then: *"You're all set. Don't forget your keys, glasses and wallet."* Ask *"where are my keys?"* and it tells you where it last saw them, or that it doesn't know.

Today it verifies stove knobs and finds your belongings; the oven display is experimental, and taps and doors are next.

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
2. **Knob positions are measured, not guessed.** SAM 3.1 finds each knob, and Hold Point measures the direction of its grip bar from the bar's straight edges, then compares it with the same knob in the user's own all-off photo. A knob turned only slightly (the gas-leak case) is never reported as off. How we got there, including three approaches that failed on real glossy knobs: [engineering notes](docs/engineering-notes.md).
3. **Every "I can't tell" becomes an instruction.** Each reason for abstaining (out of frame, covered, too dark, blurry, too far, glare) maps to a specific spoken fix, then it checks again.
4. **Short, action-first answers.** "Three off. Can't confirm back-left. Move the phone left." Danger first, never a scene description.
5. **An agent that finishes the job.** It plans the checklist from your home profile, guides you item by item, rechecks anything it abstained on, and ends with one sentence.
6. **A record you can come back to.** "Checked 9:02 PM: all off, back door locked." That answers the *"did I leave it on?"* worry after you've left.
7. **"Where are my keys?"** Once the house is verified, Hold Point closes with "You're all set. Don't forget your keys, glasses and wallet." Ask "where are my keys?" and it answers from what it has seen: an exact place when a known landmark was in view ("on the computer table, in the study, 5 minutes ago"); otherwise what it saw, clearly marked as unconfirmed ("on what looked like a brown wooden desk"). "Guide me to my keys" walks you there and says "found" only when both witnesses agree on a live view. It never invents a place (Last Seen, below).
8. **Measured honesty.** On a labelled test set built by injecting defects into real footage, we report how often Hold Point is **confidently wrong** compared with a plain AI asked the same question.

> **Headline result (20 real stove checks + 80 degraded copies):** asked "are all the burners off?", plain Muse Spark was **confidently wrong 54%** of the time, and when a knob was on it said **"all off" 64%** of the time. Hold Point was **wrong 0%** of the time: it answered when it could prove the answer, and otherwise refused with a specific fix. [Details and caveats below.](#evaluation)

---

## Beyond the front door

The leaving-home check is the first application of a general capability: **visual verification that knows its own limits**. The same engine (two independent witnesses, a deterministic decision layer and reason-coded guidance) applies wherever a wrong confident answer about a physical state has real consequences:

- **Night-time lock-up:** doors locked, windows shut, oven off before bed.
- **Reading displays nobody designed to be accessible:** oven temperature, thermostat, microwave timer, washer cycle ("It shows 375, not 350").
- **Watch mode:** "Tell me when the oven reaches 350" or "when the washer finishes."
- **Peace of mind for families:** with the user's consent, a verified check record shared with a family member, rather than a phone call asking "did you turn it off?"
- **Closing checklists for small businesses:** fryers, coffee machines and back doors at the end of a shift.
- **A safety layer for other AI agents.** Any assistant that acts on what a camera sees, on smart glasses or in a robot, needs to know when it can't see. Hold Point's abstention layer is a standalone component designed to be reused.

### Built during the event: Last Seen ("where did I leave it?")

The first use beyond appliances is already working, in `lastseen/`. With watching on, every few seconds:

- **SAM 3.1** looks for each enrolled item (keys, glasses) by name.
- **Muse Spark** checks it against the enrolment photos ("is this *your* glasses?") and names the nearest landmark from the user's own list. It can't invent a place.
- **SAM 3.1** then finds that landmark, and plain geometry on its outline checks Spark's "left of the stove".

A place is "confirmed" only when both agree. Otherwise answers say so: *"Not sure of the exact place. I saw your glasses 2 minutes ago on what looked like a wooden table, but no landmark I know was in view. Before that, they were left of the stove."* When two similar items are in view, it refuses rather than picks.

"Guide me to it" uses **Muse Wonder** to plan the search from those memories. As in the leaving-home agent, Wonder can't say "found"; only a verified view can.

Memories (a small crop, never full frames) are deleted after 24 hours, and a red WATCHING bar shows whenever frames are being sent.

Page: `/lastseen/` on the same server and passcode, for setup (items, landmarks) and watching. "Where are my keys?" and "guide me to my keys" also work hands-free on the main page. (Optional: `export HOLDPOINT_LEAVING_FIND=keys` makes finding the keys a step inside the walk-through instead.)

The capture layer targets 720p at 30 fps, matching what Ray-Ban Meta glasses stream, so the same engine can move to glasses with no change to the logic.

---

## How it works

```
                        ┌─► Witness A: Muse Spark 1.3 (structured JSON per knob, per frame) ─┐
4 phone frames ─────────┼─► Witness B: SAM 3.1 per frame → knob angle vs. OFF mark (geometry) ─┼─► decision layer ─► short spoken answer
                        └─► frame-quality checks (dark / blur / glare / too far) ─────────────┘   (no AI: agree or abstain)
                                                                                                         │
"I'm leaving, check the house" ─► Muse Wonder plans and re-checks (never states a result) ──────────────┘─► one-sentence summary
                                                                                                             + saved record + "don't forget your keys"
Last Seen (watching): SAM 3.1 finds item ─► Muse Spark: "is it YOUR keys? next to which landmark?" ─► SAM 3.1 landmark + geometry
                      ─► memory (confirmed / place unverified) ─► "where are my keys?" ─► Muse Wonder guides ─► "found" only if both agree
```

| Meta model | Role |
|---|---|
| Muse Spark 1.3 | Witness A: per-frame structured readings of each knob, display and indicator; Last Seen: confirms an object is the user's own item and names the nearest landmark |
| SAM 3.1 | Witness B: locates each knob (and later, displays) and tracks them across frames; geometry then measures each knob. Last Seen: finds items and landmarks by name |
| Muse Wonder | Leaving-home agent: plans and sequences the checks; Last Seen: plans the search for a lost item. Never states a result; final sentences are built by code |
| Muse Voice Transcribe | Voice input (hands-free commands and questions) |

**Where they run:** all four models are called through the Meta Model API (`api.meta.ai`). No other AI models are used at runtime. The decision layer, geometry, frame-quality checks and every final sentence are plain code. A local, open-weights mode (Muse Glimmer) is planned, not built.

---

## Status


Built and tested on a real stove and phone (140 automated tests):

- [x] Two witnesses + deterministic decision layer: answers only when both agree over several frames; otherwise abstains with a reason code and a spoken fix
- [x] Witness A: Muse Spark 1.3 structured per-frame readings of SAM-guided knob close-ups
- [x] Witness B: SAM 3.1 finds each knob; geometry measures its grip bar against the user's own all-off setup, with roll and tilt correction
- [x] Frame-quality checks (dark, blur, glare, too far) that veto evidence on their own
- [x] Danger-first answers: if either witness sees a knob ON, it never says "all clear"
- [x] Phone app (served from a laptop, HTTPS tunnel): hands-free voice page, manual page with setup and evaluation mode, model-calls panel
- [x] Voice in with Muse Voice Transcribe; speech out with the device's own text-to-speech
- [x] Leaving-home agent: Muse Wonder plans and re-checks; summary built from verified results only; timestamped check record
- [x] Last Seen: item and landmark setup, watching, "where are my keys?", Muse Wonder "guide me", reminder after the house check
- [x] Evaluation: 20 labelled live checks + dark/blur/far/glare copies, plain Muse Spark vs Hold Point
- [x] Spend guard (hard cap, blocks unpriced models) and response cache
- [ ] Oven display reading: built (Muse Spark + an LED-segment shape reader must agree) but experimental, so not in the walk-through
- [ ] Taps, doors, watch mode, local open-weights mode

---

## Setup

```bash
conda create -n holdpoint python=3.12 -y
conda activate holdpoint
pip install -e ".[dev]"
pytest
```

API mode needs no GPU. Set `MODEL_API_KEY`, and optionally set a spending cap with `HOLDPOINT_BUDGET_USD` (default $120). Prices are in `prices.json`; every call is logged to `.holdpoint/ledger.jsonl`, and repeated requests are served from a local cache at no cost.

### Try it on your stove

```bash
python scripts/setup_stove.py all_off.jpg --knobs 4   # once: photo with every knob OFF
python scripts/check_photo.py now.jpg                 # any time: ~1 cent per check
```

### Last Seen (keys, glasses)

```bash
python scripts/lastseen_smoke.py PHOTO_FOLDER --item glasses --phrase eyeglasses --landmark stove --room kitchen  # first test from photos, cap $0.50
uvicorn holdpoint.server:app --port 8000      # then open /lastseen/ on the phone: Setup, Start watching, Ask
# on the main page, after the leaving-home check: "where are my keys?" / "guide me to my keys"
```

### Run the phone app

```bash
export MODEL_API_KEY=...            # Meta Model API key
export HOLDPOINT_PASSCODE=...       # the phone page asks for it once
uvicorn holdpoint.server:app --port 8000
cloudflared tunnel --url http://localhost:8000   # phone cameras need HTTPS; open the printed link on the phone
```

- `/` is the hands-free page: tap Start once, then say "I'm leaving, check the house" or "where are my keys?"
- `/manual` has stove setup (all knobs OFF, from where you normally stand), single checks and evaluation mode.
- `/lastseen/` has item and landmark setup and watching.

### Reproduce the evaluation

Label live checks in evaluation mode on `/manual` (each check's frames and answer are saved to `home/live/`), then:

```bash
python scripts/run_eval.py --defects    # plain Muse Spark vs Hold Point; writes eval/results/
```

---

## Evaluation

**Question:** "Are all the burners off?" The test set is 20 live phone checks of a real 4-knob electric stove: 11 with all knobs off, and 9 with one knob on or only slightly turned. Each check also has four degraded copies (dark, blur, far, glare), for 100 cases.

**Systems compared:**
- **Plain Muse Spark:** one photo, one question, must answer.
- **Hold Point:** its live answer on clean frames; a full re-run on the degraded copies.

| | Plain Muse Spark | Hold Point |
|---|---|---|
| Real checks: correct | 9 / 20 | 11 / 20 |
| Real checks: wrong (said "all off" with a knob on, false alarm, or wrong knob) | **11 / 20** (4 unsafe "all off") | **0 / 20** |
| Real checks: refused, with a fix | can't refuse | 9 / 20 |
| Degraded copies: unsafe "all off" | 25 / 80 | **0 / 80** (refused every time) |
| All 100: confidently wrong | **54%** | **0%** |

**Read it honestly:**
- The price of 0% wrong is refusing: 45% of the real checks, mostly handheld frames where a knob was blurred or at the edge. The degraded copies are refused 100% of the time, which is the design.
- A refusal always comes with a fix. In the live app, a refusal guides the camera and checks again; how often the second try succeeds was not measured.
- This is one stove, one tester and 20 real cases. It shows the behaviour; it isn't a statistical guarantee.

Full rows: [`eval/results/rows.jsonl`](eval/results/rows.jsonl); summary: [`eval/results/summary.md`](eval/results/summary.md).

---

## Limitations

- **Not a safety-certified device.** It assists, and does not replace, the user's own checks or a smoke alarm.
- **It abstains more in poor conditions.** In bad lighting or with a shaky camera it will say what it can't see rather than guess. That is the design, but it means more prompts to adjust the camera.
- **Bar-shaped knobs look the same at OFF and half a turn.** Geometry can't tell those two apart; Hold Point only says OFF when Muse Spark independently agrees.
- **Last Seen only remembers what passes in front of the camera.** Watching stops when the phone locks. Thin or see-through items (glasses) are harder to see from far away, and places are relations in the camera view ("left of the stove"), not a map.
- **It verifies what it's set up to check.** Items need to be added to the home profile once, ideally with a sighted helper.

## Credits

All model inference uses Meta models through the Meta Model API. Other components:
- SAM 3.1 output is decoded with Meta's [`meta-sam-parser`](https://pypi.org/project/meta-sam-parser/) (SAM License).
- Image processing uses OpenCV and NumPy; the server uses FastAPI and Uvicorn.
- Speech output uses the device's built-in text-to-speech (not an AI model).
- Testing on a phone used a Cloudflare quick tunnel.
- The demo video's 21-second intro animation was drawn with Python and Cairo (illustration only; not part of the product).

All code was written by Sruthy Rajesh Bindhu (team Abstain Labs) during the event, 26 Oct - 4 Nov 2026; the git history shows the build day by day.

## Licence

[MIT](LICENSE)
