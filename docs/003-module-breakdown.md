# 003 Module Breakdown

## `amd.py` — Asterisk EAGI Client
### Purpose
Bridges Asterisk call audio and the AMD websocket inference API.

### Key Responsibilities
- AGI protocol interaction via stdin/stdout (`agi_set_variable`, `agi_get_variable`, `agi_verbose`).
- EAGI audio setup (`setup_audio_fd`) and chunk reads (`read_audio_chunk`).
- WebSocket lifecycle (`AMDWebSocketClient`).
- Result propagation to dialplan variables (`set_result_variables`, `set_human_default`).

### Interactions
- Sends binary audio and control JSON to `/ws/amd` in [server/ws_server.py](../server/ws_server.py).
- Receives final decisions and writes AGI variables consumed by downstream dialplan logic.

### Side Effects
- Writes to Asterisk channel variables.
- Performs network IO to the websocket endpoint.

### Concurrency Behavior
- Single-threaded polling loop in `main()`; non-blocking behavior depends on `select()` and websocket timeouts.

## `server/ws_server.py` — Inference Service
### Purpose
Exposes the websocket and monitoring APIs and executes the complete AMD inference pipeline.

### Key Responsibilities
- Environment-driven configuration and model/service initialization.
- Audio conversion and preprocessing (`pcm16_to_float32`, `preprocess_audio`, `normalize_audio`).
- VAD gate (`silero_has_speech`) and transcription (`transcribe_audio`).
- Classification orchestration (`classify_transcript`, `classify_ml`).
- Response construction (`build_response`) and non-blocking persistence hooks.

### Interactions
- Uses [elastic_logger.py](../elastic_logger.py) for asynchronous Elasticsearch persistence.
- Uses [audio_logger.py](../audio_logger.py) for optional SFTP audio export.

### Side Effects
- Performs model initialization and helper-worker imports at module load time.
- Mutates process-global config in `apply_runtime_config()`.

### Concurrency Behavior
- Async websocket handler with CPU/GPU work dispatched via `asyncio.to_thread()`.
- Shared global state and shared models across concurrent connections.

## `elastic_logger.py` — Elasticsearch Bulk Logger
### Purpose
Persists AMD decision events into Elasticsearch without blocking the response path.

### Key Responsibilities
- Queue log events via `log_response()`.
- Batch and flush documents in a background worker thread.
- Add derived metadata such as transcript length, decision type, and `audio_url`.
- Expose stats through `get_stats()`.

### Side Effects
- Network writes to Elasticsearch.
- Generates credential-bearing audio URLs based on storage settings.

### Concurrency Behavior
- Producer from request threads; single consumer worker with periodic bulk flush.

## `audio_logger.py` — SFTP Audio Uploader
### Purpose
Uploads decision-linked call audio to a storage box or other SFTP-compatible target.

### Key Responsibilities
- Queue audio tasks through `log_audio()`.
- Convert float audio to WAV and optionally MP3.
- Upload via Paramiko SFTP with retry logic.
- Report health and queue stats via `get_stats()`.

### Side Effects
- File creation in temporary directories during encode/upload.
- External network writes to the SFTP target.

### Concurrency Behavior
- Queue + dedicated background worker thread with reconnect and retry behavior.

## `elastic_dashboard.py` — Analytics Dashboard
### Purpose
Provides an operator-facing Streamlit dashboard over Elasticsearch decision data.

### Key Responsibilities
- Connect to Elasticsearch.
- Query indices and render charts.
- Fetch audio URLs for playback.

### Caveat
- This path introduces additional optional dependencies such as `streamlit`, `plotly`, and `requests`.

## `scripts/finetune_whisper_small.py`
### Purpose
Fine-tunes Whisper-small on custom CSV audio/transcript data.

### Key Responsibilities
- Audio loading and 2-second normalization (`load_2s`).
- Lazy dataset transform and custom collator.
- Trainer lifecycle and artifact save.

### Side Effects
- Writes checkpoints and `SAVE_OK` to the target output directory.

## `scripts/train_text_cls.py`
### Purpose
Trains the transcript classifier for `HUMAN` vs `MACHINE` labels.

### Key Responsibilities
- Text normalization.
- Word and char TF-IDF feature extraction.
- Logistic regression calibration and model serialization.

### Side Effects
- Writes the `.joblib` bundle consumed at runtime.

### Verified Risk
- The classifier is trained on `hstack([word,char])`, but the saved `vectorizer` field points only to the word vectorizer; this can leave `classifier_ready=False` in the server if feature counts do not match.

## `scripts/bulk_ws_eval.py`
### Purpose
Batch evaluation harness for websocket inference over large audio sets.

### Key Responsibilities
- Audio decode/resample/chunk conversion to PCM16.
- Parallel websocket streaming and result collection.
- CSV export with resume-mode support.

### Side Effects
- Generates network load against the inference service.
- Appends rows to the output CSV.

## `scripts/utils.py`
### Purpose
Shared helper and vocabulary assets for training and analysis tasks.

### Observed Role
- Contains large keyword lists and helper functions.
- Is not imported by the main runtime server; `ws_server.py` maintains its own keyword lists.

## `client/ws_client.py`
### Purpose
Reference client for manual protocol testing.

### Caveat
- Defaults to `ws://localhost:8080/ws/amd`, while the service is typically run on port `8989`; pass `--url` explicitly during local testing.
