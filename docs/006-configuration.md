# 006 Configuration

## Configuration Sources
- Environment variables in [server/ws_server.py](../server/ws_server.py), [amd.py](../amd.py), [elastic_logger.py](../elastic_logger.py), and [audio_logger.py](../audio_logger.py)
- Runtime websocket config messages handled by `apply_runtime_config()` in [server/ws_server.py](../server/ws_server.py)
- Script CLI arguments under [scripts](../scripts)
- Deployment values from the repository `.env` snapshot

## Server Environment Variables
Defined in [server/ws_server.py](../server/ws_server.py).

Core inference:
- `WHISPER_MODEL_DIR` default `small.en`
- `DEVICE` default `cuda`
- `COMPUTE_TYPE` default `float16`
- `CLASSIFIER_PATH` default `models/text_cls_amd_v2.joblib`
- `MAX_CALL_SEC` default `2.5`

VAD and preprocessing:
- `SILERO_VAD_DIR` default empty
- `SILERO_VAD_THRESHOLD` default `0.3`
- `SILERO_MIN_SPEECH_MS` default `100`
- `SILENCE_RMS_THRESHOLD` default `0.003`
- `TARGET_RMS` default `0.08`
- `MAX_GAIN` default `2.5`
- `SOXR_QUALITY` default `HQ`
- `INITIAL_PROMPT` default voicemail-oriented phrase list

Hallucination and gating thresholds:
- `MIN_SPEECH_RATIO` default `0.15`
- `MIN_CONFIDENCE_GREETING` default `0.50`
- `MIN_SPEECH_MS_HUMAN` default `700`
- `MIN_P2A_RATIO` default `3.5`
- `NOISE_P2A_THRESHOLD` default `3.2`
- `SHORT_SPEECH_THRESHOLD` default `500`
- `LOW_CONF_THRESHOLD` default `0.52`

Feature toggles:
- `ELASTIC_ENABLED`
- `AUDIO_SAVE_ENABLED`
- `DEBUG`

## Elasticsearch Logger Configuration
Defined in [elastic_logger.py](../elastic_logger.py).
- `ELASTICSEARCH_HOST`
- `ELASTICSEARCH_API_KEY`
- index prefix and batching/flush settings
- audio URL generation settings used by the dashboard path

## Audio Uploader Configuration
Defined in [audio_logger.py](../audio_logger.py).
- `AUDIO_SAVE_ENABLED`
- `AUDIO_FORMAT`
- `AUDIO_SAVE_FILTER`
- `STORAGE_SERVER`
- `STORAGE_USER`
- `STORAGE_PASSWORD`
- `STORAGE_PATH`

## Asterisk Client Configuration
Defined in [amd.py](../amd.py).
- `AMD_WS_URL` default `ws://127.0.0.1:8989/ws/amd`
- `AMD_CLIENT_TIMEOUT` default `5.0`

Fixed constants in `amd.py`:
- `AUDIO_FD=3`
- `SAMPLE_RATE=8000`
- `CHUNK_SIZE=3200`
- `READ_TIMEOUT=0.1`

## Runtime Config Over WebSocket
`apply_runtime_config()` currently accepts only:
- `MAX_CALL_SEC` in range `[1.0, 10.0]`
- `SILENCE_RMS_THRESHOLD` in range `[0.0, 0.05]`

## Script Configuration
- Training and batch scripts use `argparse` flags in their respective modules under [scripts](../scripts).

## Environment Differences
Development:
- Typically CPU or local GPU, local websocket URL, and optional persistence disabled.

Production (expected):
- Prebuilt model artifacts under [models](../models)
- Reachable Elasticsearch and SFTP targets if those features are enabled
- TLS termination and perimeter controls provided externally

## Configuration Risks
- Boolean parsing is strict string comparison to `true` after lowercasing.
- Runtime config modifies process globals and affects all concurrent websocket sessions.
- Service credentials and API keys are present in direct env/default form rather than through a secrets abstraction.
