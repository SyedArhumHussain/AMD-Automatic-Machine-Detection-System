# 001 Architecture

## Architectural Style
The system is an asynchronous streaming inference service with a thin telephony edge client and two non-blocking persistence integrations:
- Telephony edge client in [amd.py](../amd.py)
- FastAPI WebSocket inference service in [server/ws_server.py](../server/ws_server.py)
- Elasticsearch bulk logging in [elastic_logger.py](../elastic_logger.py)
- SFTP audio export in [audio_logger.py](../audio_logger.py)
- Offline training and evaluation utilities under [scripts](../scripts)

## Components and Responsibilities
1. **Telephony Bridge** (`amd.py`)
- Reads audio from EAGI file descriptor 3.
- Streams binary audio frames to the WebSocket server.
- Sends config handshake values such as `Number` and `VID`.
- Writes AGI variables consumed by downstream dialplan logic.

2. **WebSocket Inference Server** (`server/ws_server.py`)
- Accepts connections at `/ws/amd`.
- Collects audio and optional JSON config/flush messages.
- Executes the single-pass processing pipeline in `process_final_audio`.
- Returns final JSON containing label, confidence, transcript, timing, and audio timestamps.

3. **Classification Layer** (`server/ws_server.py`)
- Applies heuristic gating, keyword matching, and optional ML fallback through `classify_transcript` and `classify_ml`.
- Uses conservative `MACHINE` fallback behavior when confidence or evidence is weak.
- Enables the ML path only when the loaded classifier artifact passes feature-dimension validation at startup.

4. **Decision Logging Layer** (`elastic_logger.py`)
- Uses a queue-backed background worker for bulk indexing into Elasticsearch.
- Adds operational metadata such as timing, transcript length, and generated audio URLs.

5. **Audio Export Layer** (`audio_logger.py`)
- Queues audio for background SFTP upload.
- Encodes WAV/MP3 temp files and retries failed uploads.
- Applies optional storage filtering based on decision (`all`, `human`, or `machine`).

6. **Model and Training Assets** (`models`, `scripts`)
- Runtime Whisper CT2 artifact set in [models/whisper-ct2](../models/whisper-ct2).
- Fine-tuned Hugging Face style checkpoints in [models/whisper-small-finetuned](../models/whisper-small-finetuned).
- Text classifier bundle in [models/text_cls_amd_v2.joblib](../models/text_cls_amd_v2.joblib).

## Data Flow
1. Asterisk invokes [amd.py](../amd.py) via EAGI.
2. `AMDWebSocketClient.connect()` sends config JSON.
3. `read_audio_chunk()` reads PCM bytes and streams them to `/ws/amd`.
4. The server accumulates bytes into `raw_audio`.
5. On flush, timeout, or disconnect fallback, the server runs `process_final_audio()`.
6. The pipeline executes: preprocess -> VAD -> transcribe -> classify -> `build_response()`.
7. The final response returns to the AGI client; channel variables are set.
8. Optional side effects occur after the response is sent: Elasticsearch document write and SFTP audio upload.

## Control Flow
- Primary trigger: flush message (`{"type":"flush"}`) or `MAX_CALL_SEC` timeout.
- Classification order in `classify_transcript()`:
  1. Empty or hallucination guard
  2. Numeric-only checks
  3. Greeting-only checks using P2A, confidence, and speech-duration thresholds
  4. Keyword matching
  5. ML fallback (if validated and confidence threshold is met)
  6. Default `MACHINE`

## Event Flow
- WebSocket message types handled:
  - JSON config (`config` key)
  - JSON flush (`type=flush`)
  - Binary audio frames
- REST status endpoints:
  - `/health`
  - `/status`
  - `/elastic-status`
  - `/audio-status`
- The current server file does **not** define an explicit FastAPI shutdown hook; helper worker threads rely on process lifecycle.

## Internal Dependencies
- The server depends on `faster-whisper`, `torch`, `soxr`, `joblib`, and FastAPI.
- Optional services are imported at module load using `get_elastic_logger()` and `get_audio_logger()`.
- `scripts/train_text_cls.py` produces the classifier artifact consumed by startup validation in [server/ws_server.py](../server/ws_server.py).

## External Integrations
- Asterisk AGI/EAGI IO via [amd.py](../amd.py)
- Elasticsearch for decision logging via [elastic_logger.py](../elastic_logger.py)
- Hetzner or compatible SFTP storage via [audio_logger.py](../audio_logger.py)

## Architectural Tradeoffs
- Positive: low-latency streaming and deterministic single-pass decision path.
- Positive: persistence side effects are offloaded to background workers and do not block the response path.
- Tradeoff: `apply_runtime_config()` mutates process-global values, so concurrent calls can influence one another.
- Tradeoff: heavy initialization and helper worker creation happen at import/startup time.
- Tradeoff: security controls are mostly delegated to the deployment perimeter rather than the app layer.

## Model Artifact Architecture (Deep Coverage)
- The CT2 runtime directory contains tokenizer, config, and generation metadata used for inference setup.
- The fine-tuned model directory contains checkpoint snapshots (`checkpoint-*`) with trainer metadata (`trainer_state.json`) and RNG state (`rng_state.pth`).
- `SAVE_OK` is a script-level save marker emitted by [scripts/finetune_whisper_small.py](../scripts/finetune_whisper_small.py).
