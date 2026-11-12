# 014 Known Issues and Technical Debt

## Verified Fragile Areas
### 1. Global mutable runtime config
- `apply_runtime_config()` updates globals such as `MAX_CALL_SEC` and `SILENCE_RMS_THRESHOLD` in [server/ws_server.py](../server/ws_server.py).
- Concurrent connections can therefore influence each other.

### 2. Resource and lifecycle handling concerns
- Startup stderr redirection and import-time initialization increase lifecycle complexity in [server/ws_server.py](../server/ws_server.py).
- Helper workers rely on process lifetime instead of explicit coordinated shutdown.
- `scripts/finetune_whisper_small.py` writes `SAVE_OK` with direct `open(...).write(...)` usage.

### 3. Error-handling granularity
- Broad exception handling in [amd.py](../amd.py) suppresses error details in several AGI helper paths.

### 4. Performance debt
- `np.concatenate()` per incoming chunk in the websocket loop is allocation-heavy.
- Elasticsearch and audio-upload queues can defer or drop persistence work under pressure rather than applying end-to-end backpressure.

### 5. Security debt
- No built-in authentication or authorization on inference and status endpoints.
- Default client URL uses plain websocket (`ws://`).
- Hardcoded credential defaults and secret-bearing configuration files are present in the current snapshot.

### 6. Structural debt
- Very large keyword lists and mixed responsibilities are embedded directly in [server/ws_server.py](../server/ws_server.py).
- Module import side effects increase coupling and reduce testability.
- The classifier training/serving contract is inconsistent: training uses `word + char` features, while runtime validation may only see the saved word vectorizer.

## Design Flaws and Outdated Patterns
- Monolithic service file handling config, inference, protocol, and operational concerns
- Print-based operational logging mixed with background worker subsystems
- Lack of explicit schema contracts for websocket payloads

## Duplication and Abstraction Gaps
- Similar keyword concepts appear in [server/ws_server.py](../server/ws_server.py) and [scripts/utils.py](../scripts/utils.py) but are not centralized.
- Inference heuristics are hardcoded and only partly configurable through env thresholds.

## Impact Summary
- Reliability: medium risk under concurrency and high load
- Security: high risk when exposed without perimeter controls
- Maintainability: high cost due to the monolithic core and implicit coupling
