# 013 Maintenance Guide

## Safe Change Workflow
1. Review the affected runtime boundary:
- AGI bridge: [amd.py](../amd.py)
- Inference and API: [server/ws_server.py](../server/ws_server.py)
- Side services: [elastic_logger.py](../elastic_logger.py), [audio_logger.py](../audio_logger.py)

2. Preserve interface contracts:
- Websocket message formats
- AGI output variable names
- REST endpoint payload fields used by operators

3. Validate behavior with representative audio sets and the bulk evaluation script.

## Safe Extension Points
- Add new non-breaking REST endpoints in [server/ws_server.py](../server/ws_server.py).
- Extend decision metadata in `build_response()` while keeping existing fields stable.
- Add optional config keys to `apply_runtime_config()` with strict range checks.
- Add new scripts under [scripts](../scripts) without altering runtime service imports.

## Change Areas Requiring Extra Caution
- `classify_transcript()`: branch-order changes can materially alter business outcomes.
- Global runtime variables: currently process-scoped and shared across connections.
- Model-loading block at startup: failures can prevent the service from serving traffic.
- Elasticsearch and SFTP worker interactions: asynchronous side effects can hide partial failures.
- Classifier artifact format changes: the current training/serving contract is already fragile.

## Recommended Maintenance Practices
- Introduce schema definitions for websocket and AGI payload contracts.
- Add targeted unit tests before changing classification heuristics.
- Log model version and classifier hash in startup or response metadata for traceability.
- Keep training artifacts and serving artifacts versioned explicitly.

## Onboarding Checklist for New Senior Engineers
1. Read [000-system-overview.md](000-system-overview.md) and [001-architecture.md](001-architecture.md).
2. Follow [012-deployment-and-operations.md](012-deployment-and-operations.md) to run locally.
3. Validate endpoints and smoke test the websocket flow.
4. Run batch evaluation for baseline behavior.
5. Review the security and technical-debt docs before making production changes.
