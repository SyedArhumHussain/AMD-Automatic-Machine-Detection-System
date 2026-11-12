# 015 Refactor Recommendations

## Priority 1 (High Impact)
1. **Isolate per-connection config**
- Replace process-global runtime mutations with request-scoped config objects.
- Prevent cross-call interference and simplify reasoning.

2. **Split the monolithic server module**
- Extract modules for protocol, preprocessing/VAD, transcription, classification, and response schema.
- Keep [server/ws_server.py](../server/ws_server.py) as the orchestration layer.

3. **Add explicit schema contracts**
- Define message schemas for websocket config, flush, and final payloads.
- Add validation for binary chunk size and expected message types.

4. **Harden endpoint security**
- Add auth middleware or token checks for websocket and status routes.
- Enforce origin and transport requirements in the service or gateway policy.

## Priority 2 (Reliability and Performance)
1. **Replace incremental concat strategy**
- Store chunks in a list and concatenate once before processing.

2. **Improve resource lifecycle management**
- Use structured startup/shutdown lifecycle for the Elasticsearch logger, audio uploader, and descriptor cleanup.
- Remove unmanaged import-time side effects where practical.

3. **Fix the classifier artifact contract**
- Save a serving-compatible pipeline or composite vectorizer bundle from [scripts/train_text_cls.py](../scripts/train_text_cls.py).
- Make startup validation deterministic and fail loudly when the artifact is incompatible.

4. **Centralize keyword and heuristic configuration**
- Move keyword lists and thresholds into versioned config assets.
- Add checksum or version telemetry in responses and logs.

5. **Improve error taxonomy**
- Replace broad catches with typed exception handling and structured error codes.

## Priority 3 (Developer Experience)
1. **Add a test suite and CI pipeline**
- Unit tests for utility and decision logic.
- Integration tests for websocket protocol and AGI path.

2. **Add packaging and deployment artifacts**
- Dockerfile and sanitized environment templates.
- Basic CI checks for lint, test, and build.

3. **Improve operational observability**
- Structured JSON logs with correlation IDs.
- Metric export for queue depth, decision-method rates, and latency distribution.

4. **Externalize secrets and URL generation**
- Remove hardcoded service defaults and credential-bearing audio URLs.
- Move secret handling to a managed configuration source.

## Suggested Refactor Sequence
1. Schema and config isolation.
2. Artifact-contract fix plus behavior-preserving tests.
3. Module decomposition.
4. Security, observability, and performance hardening.
