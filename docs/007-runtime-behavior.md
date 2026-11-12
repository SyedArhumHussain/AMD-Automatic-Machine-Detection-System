# 007 Runtime Behavior

## Startup Sequence (Server)
1. Module import performs stderr redirection and warning suppression in [server/ws_server.py](../server/ws_server.py).
2. Feature toggles and thresholds are loaded from the environment.
3. Whisper is initialized (`WhisperModel`).
4. Silero VAD is loaded via `_load_silero_vad()` with local/hub fallback.
5. The optional classifier bundle is loaded and validated.
6. Optional Elasticsearch and SFTP helper services are initialized.
7. FastAPI starts with websocket and REST routes.

## Startup Sequence (Asterisk Client)
1. `main()` in [amd.py](../amd.py) reads the AGI environment.
2. `setup_audio_fd()` validates and configures non-blocking FD 3.
3. `AMDWebSocketClient.connect()` opens the websocket and sends config.
4. The audio loop starts reading chunks and sending binary messages.

## Request Lifecycle
1. The websocket is accepted in `ws_amd()`.
2. The server receives config and replies with `config_ack`.
3. Binary audio frames are appended to `raw_audio`.
4. On flush, timeout, or disconnect fallback, `process_final_audio()` runs through `asyncio.to_thread()`.
5. The decision payload is sent to the client.
6. Optional side effects then occur: Elasticsearch logging and SFTP audio upload.

## Inference Pipeline Lifecycle
1. Pre-checks: no audio or too-short audio -> immediate `MACHINE`
2. Preprocess: normalization and resampling to 16kHz
3. Silence gate on RMS
4. Silero VAD speech gate
5. Whisper transcription with fallback mode if needed
6. Transcript classification with heuristic + keyword + ML path
7. Final response construction

## Background Tasks and Workers
- Elasticsearch logger starts a daemon worker thread at initialization.
- Audio uploader starts a daemon worker thread with reconnect/retry behavior.
- Websocket processing invokes blocking compute via thread offload.

## Shutdown Behavior
- The current FastAPI app does **not** define a dedicated shutdown hook for the helper workers.
- In the websocket `finally` block, the server still attempts fallback decision emission if none was sent.

## Operational Timing
- The websocket collect loop uses `MAX_CALL_SEC` as a hard stop.
- AGI client timeout is controlled by `AMD_CLIENT_TIMEOUT`.
- The bulk evaluation script controls per-file timeout via `--timeout_s`.

## Hidden Dependency Behaviors
- Importing [server/ws_server.py](../server/ws_server.py) executes model and helper-service initialization immediately.
- If classifier validation fails at startup, the service continues in heuristic/keyword-only mode.
