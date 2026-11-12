# 011 Testing Strategy

## Current State
No automated test suite (unit, integration, or e2e) is present in this repository.

## Existing Validation Mechanisms
- Manual websocket smoke testing via [client/ws_client.py](../client/ws_client.py)
- Bulk inference evaluation via [scripts/bulk_ws_eval.py](../scripts/bulk_ws_eval.py)
- Training-time metric output in [scripts/train_text_cls.py](../scripts/train_text_cls.py)

## Recommended Test Architecture
### 1. Unit tests
- Transcript utilities: `normalize_text`, greeting detectors, numeric detectors
- Decision orchestration: `classify_transcript()` branch coverage
- Response schema and probability consistency in `build_response()`

### 2. Component tests
- Websocket handler behavior for config/audio/flush/fallback paths
- Elasticsearch logger queue and flush semantics with the client mocked
- Audio uploader retry and filtering behavior with temp files and mocked SFTP

### 3. Integration tests
- End-to-end AGI-like flow from synthetic PCM input to final AMD variables
- Websocket streaming tests with realistic chunk sizes and timeouts
- Optional persistence-path tests with Elasticsearch and SFTP stubs

### 4. Regression datasets
- Curated audio/transcript sets for edge classes:
  - silence/noise
  - short greetings
  - IVR numeric menus
  - voicemail prompts
  - ambiguous transcripts

## Coverage Gaps (Verified)
- No test guard for global config mutation side effects
- No schema contract tests for websocket payloads
- No stress/load test committed for concurrent connection behavior
- No negative/security tests for oversized frames or malformed payload fuzzing
- No regression test protecting the current classifier artifact compatibility issue (`word + char` training vs runtime feature validation)

## Mocking Approach Recommendation
- Mock `WhisperModel.transcribe()` and Silero helpers for deterministic unit tests.
- Mock Elasticsearch writes in logger-worker tests.
- Mock Paramiko SFTP and audio encode paths in uploader tests.

## CI Expectation
The repository currently has no CI configuration. Add lint + unit + integration checks before production promotion.
