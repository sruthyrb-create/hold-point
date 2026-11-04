"""Oven display check: is the oven off (display shows the clock) or on (temperature,
"PrE", "bAkE"...)? Two independent witnesses, like the stove.

Witness A (Muse Spark) reads the display and says what it shows.

Witness B (plain code, no AI) finds the lit LED segments by colour and brightness and
reads the display's SHAPE, not its digits:
    clock    a row of digit-sized glyphs with a colon (two small dots stacked) -> "off"
    cooking  a row of 2-4 digit-sized glyphs and no colon (e.g. "350", "PrE")  -> "on"
    other    nothing lit, one glyph, a jumble                                    -> abstain
A colon may blink, so a frame without one is only "cooking" if no frame of the
check shows a colon (see read_display_frames).

Known limits (stated in the README): a kitchen TIMER also shows a colon, so a timer
counting down while the oven is on would read "clock" to Witness B; Witness A is told
to report timers separately, which makes the two disagree and Hold Point abstain.
"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from .decision import DecisionConfig, decide
from .meta_client import MetaClient, encode_frame, prepare_frame
from .quality import frame_reasons
from .schemas import FrameReadings, ItemReading, ReasonCode

ITEM = "oven"
NAME = "the oven"
SPARK = "muse-spark-1.3"

MIN_V = 120          # LED segments are bright...
MIN_S = 70           # ...and coloured (white glare is not a lit segment)
LED_HUE = (30, 95)   # OpenCV hue range of the LEDs: green to cyan (Sruthy's display measured 40-45).
                     # A red display would be (0, 10); set "led_hue" in home/profile.json to change it.
MIN_DIGIT_PX = 20    # digits shorter than this in the frame = too far (13 px on 31 Oct: colon unreadable)
READ_HEIGHT = 80     # the display is enlarged so its digits are about this tall before reading


def led_mask(frame_bgr: np.ndarray, hue: tuple[int, int] = LED_HUE) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    m = (hsv[..., 2] >= MIN_V) & (hsv[..., 1] >= MIN_S) & (h >= hue[0]) & (h <= hue[1])
    return m.astype(np.uint8) * 255


def locate_display(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """The display = the densest cluster of LED-coloured pixels. Returns (x0, y0, x1, y1).
    Two passes: a small grouping finds the brightest glyph cluster and its height; a second
    grouping sized to that height (characters sit about one height apart) joins the whole line."""
    if mask.sum() < 255 * 6:
        return None

    def biggest(m, kh, kw):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((kh, kw), np.uint8)))
        if n <= 1:
            return None
        best = max(range(1, n), key=lambda i: int(m[labels == i].sum()))
        ys, xs = np.nonzero((labels == best) & (m > 0))
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    first = biggest(mask, 3, 3)
    if first is None:
        return None
    h = max(3, first[3] - first[1])
    return biggest(mask, max(3, h // 3), max(3, int(h * 1.5)))


def _glyphs(mask: np.ndarray, digit_h: int) -> list[tuple[int, int, int, int]]:
    """Boxes (x, y, w, h) of glyphs. Only a VERTICAL close joins a digit's segments, so
    neighbouring digits and the colon dots stay separate."""
    k = max(3, int(digit_h * 0.2))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    return [tuple(int(v) for v in stats[i, :4]) for i in range(1, n) if stats[i, 4] >= 6]


def read_led(frame_bgr: np.ndarray, hue: tuple[int, int] = LED_HUE) -> dict:
    """{'pattern': 'clock'|'cooking'|None, 'digits': n, 'colon': bool, 'boxes': [...], 'reason': ReasonCode}
    boxes are in the frame's own pixel coordinates."""
    none = {"pattern": None, "digits": 0, "colon": False, "boxes": [], "reason": ReasonCode.OUT_OF_FRAME}
    box = locate_display(led_mask(frame_bgr, hue))
    if box is None:
        return none
    x0, y0, x1, y1 = box
    if y1 - y0 < MIN_DIGIT_PX:
        return {**none, "reason": ReasonCode.TOO_FAR}
    pad = max(2, (y1 - y0) // 2)
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(frame_bgr.shape[1], x1 + pad), min(frame_bgr.shape[0], y1 + pad)
    scale = min(8.0, max(1.0, READ_HEIGHT / (y1 - y0)))
    crop = cv2.resize(frame_bgr[Y0:Y1, X0:X1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    mask = led_mask(crop, hue)
    boxes = _glyphs(mask, int((y1 - y0) * scale))
    if not boxes:
        return {**none, "reason": ReasonCode.LOW_EVIDENCE}
    tallest = max(h for _, _, _, h in boxes)
    digits = [b for b in boxes if b[3] >= 0.6 * tallest]
    ref = max(digits, key=lambda b: b[3])
    cy = ref[1] + ref[3] / 2
    digits = [b for b in digits if abs(b[1] + b[3] / 2 - cy) < 0.5 * tallest]
    # Colon dots are found on the UNJOINED mask: the vertical join would merge the two dots.
    n_raw, _, raw, _ = cv2.connectedComponentsWithStats(mask)
    dots = [tuple(int(v) for v in raw[i, :4]) for i in range(1, n_raw)
            if raw[i, 4] >= 4 and raw[i, 3] <= 0.3 * tallest and raw[i, 2] <= 0.4 * tallest]
    dx0, dx1 = min(b[0] for b in digits), max(b[0] + b[2] for b in digits)
    inline = [d for d in dots if dx0 < d[0] + d[2] / 2 < dx1 and abs(d[1] + d[3] / 2 - cy) < 0.6 * tallest]
    colon = any(abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) < 0.3 * tallest and (a[1] + a[3] / 2) < cy < (b[1] + b[3] / 2)
                for a in inline for b in inline if a is not b)
    n = len(digits)
    pattern = "clock" if colon and 3 <= n <= 4 else "cooking" if not colon and 2 <= n <= 4 else None
    back = lambda b: (int(X0 + b[0] / scale), int(Y0 + b[1] / scale), max(1, int(b[2] / scale)), max(1, int(b[3] / scale)))
    return {"pattern": pattern, "digits": n, "colon": colon, "boxes": [back(b) for b in digits + inline],
            "reason": ReasonCode.OK if pattern else ReasonCode.LOW_EVIDENCE}


def read_display_frames(frames: list[np.ndarray]) -> list[FrameReadings]:
    """Witness B over a check's frames. A colon seen in ANY frame means the display is in clock
    mode (colons blink), so colon-less frames of a clock check are not counted as 'cooking'."""
    reads = [read_led(f) for f in frames]
    any_colon = any(r["colon"] for r in reads)
    out = []
    for i, r in enumerate(reads):
        pattern = r["pattern"]
        if pattern == "cooking" and any_colon:
            pattern = "clock" if r["digits"] in (3, 4) else None   # blinking colon caught "off"
        if pattern is None:
            reading = ItemReading(item_id=ITEM, visible=False, reason_code=r["reason"] if r["reason"] is not ReasonCode.OK else ReasonCode.LOW_EVIDENCE)
        else:
            reading = ItemReading(item_id=ITEM, visible=True, state="off" if pattern == "clock" else "on", confidence=1.0)
        out.append(FrameReadings(frame_index=i, witness="geometry", items=[reading]))
    return out, reads


# ---------------------------------------------------------------- Witness A: Muse Spark

SYSTEM = ("You read appliance displays for a blind user. Never guess. If the display is not clearly "
          "readable in a frame, say unreadable.")
SCHEMA = {"name": "OvenDisplay", "schema": {"type": "object", "additionalProperties": False, "required": ["frames"],
    "properties": {"frames": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["frame", "shows", "text", "confidence"],
        "properties": {"frame": {"type": "integer"},
                       "shows": {"type": "string", "enum": ["clock", "cooking", "timer", "blank", "unreadable"]},
                       "text": {"type": ["string", "null"]},
                       "confidence": {"type": "number"}}}}}}}
PROMPT = ("There are {n} frames of an oven's digital display, numbered 0 to {last}. For EACH frame say what the "
          "display shows: 'clock' (the time of day, e.g. 10:16), 'cooking' (a temperature such as 350, or a "
          "cooking word such as PrE, BAKE, bAKE, Hot, CLn), 'timer' (a kitchen timer counting down), 'blank', "
          "or 'unreadable'. Copy the characters you see into text. confidence is 0 to 1.")
STATE = {"clock": "off", "cooking": "on"}


def read_spark(client: MetaClient, frames: list[np.ndarray], tag: str = "oven") -> list[FrameReadings]:
    raw = client.chat_json(SPARK, SYSTEM, PROMPT.format(n=len(frames), last=len(frames) - 1),
                           [encode_frame(f) for f in frames], SCHEMA, reasoning_effort="low", tag=tag)
    out = []
    for fr in raw.get("frames", []):
        i = fr.get("frame")
        if not isinstance(i, int) or not 0 <= i < len(frames):
            continue
        state = STATE.get(fr.get("shows"))
        conf = min(max(float(fr.get("confidence") or 0), 0.0), 1.0)
        if state:
            r = ItemReading(item_id=ITEM, visible=True, state=state, confidence=conf)
        else:   # a timer is NOT proof the oven is off; blank/unreadable is no evidence
            r = ItemReading(item_id=ITEM, visible=False, confidence=conf,
                            reason_code=ReasonCode.WITNESS_DISAGREE if fr.get("shows") == "timer" else ReasonCode.LOW_EVIDENCE)
        out.append(FrameReadings(frame_index=i, witness="vlm", items=[r]))
    return out, raw


# ---------------------------------------------------------------- the check

def answer_for(v) -> str:
    if v.answered and v.state == "off":
        return "Yes, the oven is off. The display shows the clock."
    if v.answered:
        return "No. The oven is ON. Turn it off."
    if v.possibly_on:
        return "Not all clear. The oven may be ON. Check it."
    fix = {ReasonCode.TOO_DARK: "Turn on a light.", ReasonCode.TOO_FAR: "Move closer to the display.",
           ReasonCode.GLARE: "Tilt the phone slightly.", ReasonCode.BLUR: "Hold still.",
           ReasonCode.OUT_OF_FRAME: "Point the phone at the oven display."}.get(v.reason_code, "Move closer to the display and hold still.")
    return f"Can't confirm the oven. {fix}"


def _overlay(frame: np.ndarray, boxes: list) -> str:
    img = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 200, 60), 2)
    if img.shape[1] > 720:
        img = cv2.resize(img, (720, int(img.shape[0] * 720 / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def run_oven_check(client: MetaClient, frames: list[np.ndarray],
                   cfg: DecisionConfig = DecisionConfig(), tag: str = "live") -> dict:
    start = time.perf_counter()
    spent = client.ledger.spent()
    frames = [prepare_frame(f) for f in frames]
    # A display is a dark panel with glowing digits: "too dark" and "glare" checks don't apply to it.
    # Darkness shows up instead as "no lit digits found"; blur still vetoes a frame.
    quality = {i: [q for q in frame_reasons(f) if q is ReasonCode.BLUR] for i, f in enumerate(frames)}
    geo, led = read_display_frames(frames)
    calls, errors, vlm, raw = [], [], [], {}
    t = time.perf_counter()
    try:
        vlm, raw = read_spark(client, frames, tag=tag)
        calls.append({"model": SPARK, "job": f"read the oven display in {len(frames)} frames (one request)",
                      "seconds": round(time.perf_counter() - t, 2), "result": f"{len(vlm)} readings"})
    except Exception as e:
        if type(e).__name__ == "BudgetExceeded":
            raise
        errors.append(f"{SPARK}: {type(e).__name__}: {e}"[:200])
    v = decide([ITEM], vlm, geo, quality={i: q for i, q in quality.items() if q}, cfg=cfg)[0]
    answer = "Can't check right now. Try again in a moment." if errors else answer_for(v)
    spark_reads = [{"frame": f.frame_index + 1, "state": r.state, "confidence": r.confidence,
                    "reason": r.reason_code.value} for f in vlm for r in f.items]
    led_reads = [{"frame": i + 1, "state": g.items[0].state, "turn_deg": None,
                  "reason": g.items[0].reason_code.value, "digits": led[i]["digits"], "colon": led[i]["colon"]}
                 for i, g in enumerate(geo)]
    best = max(range(len(frames)), key=lambda i: len(led[i]["boxes"])) if frames else 0
    return {
        "item": ITEM,
        "answer": answer,
        "all_clear": not errors and v.answered and v.state == "off",
        "danger": (v.answered and v.state != "off") or v.possibly_on,
        "guidance": None if v.answered or v.possibly_on else answer.split(". ", 1)[-1],
        "items": [{"id": ITEM, "name": "oven display", "spark": spark_reads, "sam": led_reads, "sam_middle_deg": None,
                   "verdict": {"answered": v.answered, "state": v.state, "reason": v.reason_code.value,
                               "possibly_on": v.possibly_on, "evidence_frames": v.evidence_frames}}],
        "spark_text": [f.get("text") for f in raw.get("frames", [])] if raw else [],
        "calls": calls + [{"model": "LED shape reader (no AI)", "job": "find lit segments; clock has a colon",
                           "seconds": 0, "result": ", ".join(f"{r['digits']}d{':' if r['colon'] else ''}" for r in led)}],
        "errors": errors,
        "quality": {str(i + 1): [q.value for q in qs] for i, qs in quality.items() if qs},
        "frames": len(frames),
        "seconds": round(time.perf_counter() - start, 2),
        "cost_usd": round(client.ledger.spent() - spent, 4),
        "overlay": _overlay(frames[best], led[best]["boxes"]) if frames else "",
    }
