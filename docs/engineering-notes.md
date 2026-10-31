# Engineering notes: measuring a real stove knob

How Witness B (geometry) reached its current design, tested on a real electric coil stove
with four glossy bar-grip knobs, photographed by hand from three different spots.

## Test photos

| Photo | Ground truth |
|---|---|
| all off (setup) | all four knobs OFF |
| `on` | knob 3 turned a long way; the rest OFF |
| `slight` | knob 1 turned slightly (the "gas-leak" case); the rest OFF |

Muse Spark 1.3 on its own (Witness A) said **all four knobs were ON** in `on`, at
0.85-0.95 confidence. Only one was. This is the failure Hold Point exists to catch.

## Attempt 1: rotation matching (dropped)

Rotate the knob's OFF reference crop and find the best-matching angle.
Worked on synthetic knobs. On the real stove the turned knob failed to match at all,
and an untouched knob drifted 10 degrees. Cause: the knob's chrome sides reflect
differently every time the knob turns or the camera moves, so pixel comparison mostly
matched reflections.

## Attempt 2: SAM 3.1 outlines the grip bar (not possible)

Tried four prompts ("knob handle", "black bar on knob", "knob grip", "knob pointer")
on all three photos (12 calls, 3 cents). SAM returns the whole knob or nothing; it does
not separate the bar.

## Attempt 3: pointer tip from the knob's outline (dropped)

When a knob turns, its pointer tip sticks out past the round edge (knob 3's box grew from
80 to 96 px wide). But with every knob OFF, the outlines already bulge 5-11 px at the
bottom: the knob's side, seen from above. That bulge depends on where you stand and is
as large as the tip signal.

## Attempt 4: darkest region / darkest line (dropped)

The dark stripe is the bar's side face and sits off-centre (it's a raised fin seen at an
angle), and the rim's shadow is also dark. Untouched knobs read anywhere from 2 to 46 degrees.

## What works: dominant edge direction

The bar's two long, straight edges all point one way, while the round rim's edges point
every way. Inside each SAM-located knob we:

1. boost local contrast (CLAHE) and remove gradual shading across the dome, so a
   lighting gradient can't pose as an edge;
2. build a histogram of edge directions weighted by edge strength;
3. take the peak as the bar's direction, and require a clear peak (plain dome: 1.7;
   real bars: 3.8-5.6; threshold 2.8);
4. compare with the same knob's direction in the user's all-off setup photo.

Results on the real photos (turn from each knob's own OFF direction):

| Photo | Knob 1 | Knob 2 | Knob 3 | Knob 4 |
|---|---|---|---|---|
| `on` | -1.8 off | +1.1 off | **+80.8 ON** | -1.7 off |
| `slight` | **+13.6 unsure** | -3.0 off | -4.9 off | -2.6 off |

Untouched knobs stay within +/-5 degrees across different camera positions, so the OFF
window is +/-6 degrees and 6-14 degrees is "unsure" (abstain). The slight turn lands in
"unsure", so Hold Point never calls it off.

Combined with Witness A, Hold Point's answers become:
- `on`: "No. Knob 3 is ON." (Witness A's false ONs for knobs 1, 2 and 4 are not repeated)
- `slight`: "Not all clear. Knob 1 may be ON. Check it."

## Known limit

A straight bar looks the same pointing up (OFF) and straight down (half a turn), so
geometry measures direction modulo 180 degrees. Hold Point only says OFF when Witness A
independently agrees, and the evaluation reports how often that case occurs.
