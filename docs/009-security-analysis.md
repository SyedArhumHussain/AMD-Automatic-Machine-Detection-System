# 009 Security Analysis

## Security Posture Summary
The repository contains no built-in authentication or authorization layer for websocket inference. Security controls are assumed to be provided by external network boundaries.

## Authentication and Authorization
Verified from [server/ws_server.py](../server/ws_server.py):
- `/ws/amd` accepts websocket connections without token, API-key, or session checks.
- REST status endpoints are unauthenticated.

Verified from [amd.py](../amd.py):
- The client connects directly to the configured websocket URL; no auth headers or credentials are sent.

## Transport Security
Verified from defaults:
- `AMD_WS_URL` uses `ws://` in [amd.py](../amd.py).
- Sample and script usage likewise use unencrypted websocket URLs.

Impact:
- Traffic confidentiality and integrity depend on external TLS termination or private network segmentation.

## Input Validation
Verified behaviors:
- JSON parsing is guarded with `try/except JSONDecodeError`.
- Runtime config keys are allow-listed and clamped.
- No explicit maximum size check is applied per binary websocket frame.

## Secret Handling
- Service secrets are passed directly via env variables and also have hardcoded fallbacks in helper modules.
- The workspace includes a `.env` file with deployment-specific values.
- The repository does not include a dedicated secret-management integration.

## Data Protection
- Optional persisted call artifacts include transcripts, metadata, and uploaded audio.
- Elasticsearch logs include the decision payload and transcript text.
- Generated `audio_url` fields can expose storage access patterns and should be treated as sensitive.

## Attack Surfaces
1. Websocket ingress `/ws/amd` (binary + JSON)
2. Status endpoints exposing configuration and readiness signals
3. Elasticsearch logging path if reachable without proper network controls
4. SFTP storage path and generated audio URLs

## Verified Risks
1. Unauthenticated websocket and status endpoints
2. Unencrypted default websocket URL configuration
3. `apply_runtime_config()` mutates process-global thresholds from client-supplied config
4. `skip_utf8_validation=True` in websocket client construction in [amd.py](../amd.py)
5. Hardcoded credential defaults and secret-bearing `.env` content raise exposure risk

## Hardening Recommendations
- Enforce TLS (`wss`) at edge and internal links where applicable.
- Add connection authentication and origin checks in the websocket handler.
- Move runtime overrides to per-connection state rather than process globals.
- Add max payload and frame validation for websocket binary data.
- Restrict status endpoints behind internal auth or network policy.
- Remove committed secrets, rotate credentials, and define retention/encryption policy for stored audio and indexed payloads.
