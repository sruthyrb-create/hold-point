# Day 1 findings (29 Oct 2026), checked against dev.meta.ai docs

## Corrections to the build brief

| Brief said | Docs say | Effect |
|---|---|---|
| Voice Transcribe is realtime-only at `/v1/realtime` | Realtime WebSocket is `wss://api.meta.ai/v1/asr/realtime`; there is also a file endpoint `POST /v1/asr/transcribe` | Use the correct path. The file endpoint makes voice testable from recordings. |
| Local mode: "SAM 3.1 and a 30B model" via Spark Open or Glimmer | Muse Spark Open is an **816B-parameter MoE (42B active)**. Muse Glimmer is the **30B dense** one (text + image in, Apache 2.0) | Spark Open can't be self-hosted on $100 of Modal. Local mode should be **Glimmer + self-hosted SAM 3.1**; Spark Open only via a hosted provider (Baseten). |
| Open question: does Spark 1.3 take video? | Yes: text, image, video, audio, PDF | Still send sampled frames for per-frame witness readings (cheaper, gives the frame spread the decision layer needs). |

## Confirmed

- Base URL `https://api.meta.ai/v1`, key in `MODEL_API_KEY`, OpenAI SDK works (pass `api_key` explicitly; it does not read `MODEL_API_KEY` on its own).
- Model IDs: `muse-spark-1.3`, `muse-wonder-1.0`, `sam-3.1`, `muse-voice-transcribe-1.0`.
- Structured output: `response_format={"type":"json_schema",...}` on Chat Completions (`text.format` on Responses). Decoding is constrained, works with image input. Our schema is `VLM_RESPONSE_SCHEMA` in `holdpoint/schemas.py`.
- Images: base64 data URLs fine, up to 50 per request. A ~1280px image costs ~1,300-1,500 input tokens, so 720p frames are cheaper.
- Spark returns reasoning tokens by default (a trivial question used ~240). **Set a low reasoning effort for per-frame checks** or the $150 goes fast.
- SAM 3.1 over the Responses API: **one noun phrase per request** ("send several requests" for several object types). Also accepts box/point prompts, video with tracking, and can return confidence values. Output is special-token text that needs parsing (see "Reading segmentation output" doc).
- Glimmer licence: Apache 2.0. Spark Open licence: not checked yet.

## Design note: keep the witnesses independent

Muse Spark also does "perception grounding" (boxes on a 0-1000 grid). Don't use Spark's boxes to tell SAM where the knobs are, or the two witnesses stop being independent. SAM gets text prompts (or boxes from the home profile captured at setup), never from Spark.

## Still to check

- SAM output format and per-request limits/pricing (read `docs/sam/reading-segmentation` and pricing page).
- Muse Spark Open licence and whether Baseten's `muse-spark-open-1026` takes images.
- Official rules questions in Discord #questions.
