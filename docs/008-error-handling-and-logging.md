# 008 Error Handling and Logging

## Error Categories
### 1. Input and protocol errors
- Malformed websocket JSON is ignored in the websocket handler.
- Empty or invalid audio falls through to conservative `MACHINE` response paths.

### 2. Model and inference errors
- VAD and ASR functions catch exceptions and apply safe fallbacks.
- ML classifier errors or feature mismatches fall back to heuristic behavior and default `MACHINE` handling.

### 3. Integration errors
- AGI IO errors in [amd.py](../amd.py) use broad exception handling.
- [elastic_logger.py](../elastic_logger.py) catches bulk-index failures and increments internal counters.
- [audio_logger.py](../audio_logger.py) catches queue, encode, connection, and upload failures and records them in stats.

## Retry and Fallback Strategy
- VAD unavailable: allow the transcription path to continue.
- Transcription failure: empty transcript path maps to `MACHINE`.
- Websocket disconnect fallback: final processing is attempted in `finally`.
- AGI client fallback: `set_human_default()` on connection or processing failure.
- Audio upload retry behavior is internal to the SFTP worker and does not block the AMD response.

## Logging Behavior
- The server uses `print` for operational logs in [server/ws_server.py](../server/ws_server.py).
- The Elasticsearch logger tracks counters and errors in-memory.
- The audio uploader uses Python logging and health counters.

## Monitoring Hooks
- `/health`: high-level ready state
- `/status`: thresholds and feature-state detail
- `/elastic-status`: Elasticsearch queue and success/failure counts
- `/audio-status`: audio uploader queue and connection metrics

## Alert Triggers (Recommended Operationally)
The code does not define alerting. Practical trigger points based on available metrics:
- Elasticsearch failed-count growth
- Audio queue saturation near its max size
- Rising fallback methods (`empty_or_hallucination`, `silence`, `no_audio`)
- Frequent websocket timeout/disconnect before flush

## Verified Weaknesses
- Warning/log suppression is enabled at startup (`PYTHONWARNINGS=ignore`, etc.).
- Multiple broad exception handlers in [amd.py](../amd.py) hide fault granularity.
- Queue-based side services can drop or defer operational signals without upstream backpressure.
