# 000 System Overview

## Business Purpose
This system performs Answering Machine Detection (AMD) for outbound telephony workflows by classifying early-call audio as `HUMAN` or `MACHINE`. It is designed for rapid call-routing decisions in dialer and IVR environments.

## Problem It Solves
Purely acoustic AMD is often brittle. This implementation combines speech-to-text, heuristic transcript analysis, keyword matching, and an optional ML fallback to improve detection quality while keeping latency low.

## Target Users
- Telephony platform engineers integrating with Asterisk EAGI.
- Applied AI and ASR engineers maintaining Whisper and classifier artifacts.
- Operations/SRE teams running WebSocket inference services and supporting persistence systems.

## Core Capabilities
- Real-time binary audio ingestion over WebSocket at `/ws/amd` in [server/ws_server.py](../server/ws_server.py).
- Single-pass inference pipeline in `process_final_audio` in [server/ws_server.py](../server/ws_server.py).
- Asterisk AGI integration in [amd.py](../amd.py) that sets `AMDSTATUS`, `AMDCONFIDENCE`, `AMDTRANSCRIPT`, and related channel variables.
- Optional bulk decision logging to Elasticsearch through [elastic_logger.py](../elastic_logger.py).
- Optional SFTP audio export and MP3/WAV handling through [audio_logger.py](../audio_logger.py).
- Training and evaluation utilities under [scripts](../scripts).

## System Boundaries
In scope:
- Audio ingest, preprocessing, VAD gating, transcription, transcript classification, and decision response.
- Operational persistence side services (Elasticsearch logging and SFTP audio export).
- Offline model training and evaluation workflows.

Out of scope:
- Dialer campaign orchestration beyond the returned AMD decision.
- Edge TLS termination, perimeter authentication, and firewall policy.
- CI/CD or infrastructure-as-code orchestration (not committed in this repository snapshot).

## Assumptions
- `amd.py` receives PCM16 mono 8kHz audio from Asterisk on EAGI FD 3.
- The server accepts binary chunks plus JSON config/flush messages in [server/ws_server.py](../server/ws_server.py).
- Runtime model artifacts are present under [models](../models).

## Constraints
- Default `MAX_CALL_SEC` is `2.5` seconds in [server/ws_server.py](../server/ws_server.py).
- Runtime behavior is heavily environment-variable driven.
- No built-in auth/origin enforcement exists on the public service endpoints.
- No automated test suite is currently committed.

## Non-Goals
- General-purpose transcription service.
- Multi-tenant identity or authorization framework.
- Complete observability stack with tracing and alert automation.

## Verified Implementation Evidence
- FastAPI app and endpoints in [server/ws_server.py](../server/ws_server.py).
- AGI interaction and channel-variable writes in [amd.py](../amd.py).
- Elasticsearch/audio-save feature toggles in [server/ws_server.py](../server/ws_server.py).
- Dependency stack in [requirements.txt](../requirements.txt).
