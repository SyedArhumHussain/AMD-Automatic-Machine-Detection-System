# 004 API and Interface Reference

## WebSocket API
Endpoint: `/ws/amd` implemented in [server/ws_server.py](../server/ws_server.py).

## Inbound Message Types
### 1. JSON config handshake
```json
{
  "config": {
    "Number": "string",
    "VID": "string",
    "MAX_CALL_SEC": 2.5,
    "SILENCE_RMS_THRESHOLD": 0.003
  }
}
```
Behavior:
- `Number` and `VID` are extracted for call context.
- Only the runtime keys allow-listed in `apply_runtime_config()` are accepted.
- Unknown keys are ignored.

### 2. Binary audio frame
- Type: raw bytes interpreted as PCM16 samples by `pcm16_to_float32()`.
- The Asterisk path assumes 8kHz PCM audio; example clients may stream 16kHz WAV-derived chunks for local testing, so verify sample-rate expectations in your deployment.

### 3. Flush control message
```json
{"type":"flush"}
```
Behavior:
- Triggers final processing and response generation immediately.

## Outbound Message Types
### 1. Config acknowledgement
```json
{
  "type": "config_ack",
  "applied": {"MAX_CALL_SEC": 2.8},
  "number": "string"
}
```

### 2. Final decision
```json
{
  "type": "final",
  "label": "HUMAN|MACHINE",
  "decision": "HUMAN|MACHINE",
  "confidence": 0.0,
  "proba_human": 0.0,
  "proba_machine": 0.0,
  "transcript": "string",
  "method": "string",
  "elapsed_ms": 0,
  "number": "string",
  "audio_start_time": "[Apr 06 10:12:13]",
  "audio_start_timestamp": "2026-04-06T10:12:13.000000",
  "audio_end_time": "[Apr 06 10:12:15]",
  "audio_end_timestamp": "2026-04-06T10:12:15.000000",
  "audio_duration_sec": 2.0
}
```

## Validation and Error Conditions
- Malformed JSON is ignored via the `JSONDecodeError` path in the websocket handler.
- Empty or too-short audio yields `MACHINE` decisions such as `no_audio` or `too_short`.
- Silence threshold failures yield `MACHINE` via the `silence` path.
- If the connection closes before an explicit flush, the server attempts final processing in the `finally` block.

## REST Endpoints
Implemented in [server/ws_server.py](../server/ws_server.py).

1. `GET /health`
- Returns readiness, feature flags, and threshold summary.

2. `GET /status`
- Returns expanded runtime config, feature-state detail, and keyword counts.

3. `GET /elastic-status`
- Returns Elasticsearch logger status and counters.

4. `GET /audio-status`
- Returns SFTP audio uploader status and queue/connection metrics.

## AGI Interface Contract
Implemented in [amd.py](../amd.py).

Inputs:
- AGI environment from stdin (`read_agi_env()`)
- PCM audio from FD 3 (`read_audio_chunk()`)

Outputs:
- `AMDSTATUS` (`HUMAN|MACHINE`)
- `AMDCAUSE`
- `AMDTRANSCRIPT`
- `AMDELAPSED`
- `AMDMETHOD`
- `AMDCONFIDENCE`
- Optional probabilities: `AMDPROBAHUMAN`, `AMDPROBAMACHINE`

## CLI Interfaces
1. [scripts/finetune_whisper_small.py](../scripts/finetune_whisper_small.py)
- `--csv`, `--out_dir`, `--base_model`, `--lang`, `--epochs`, `--batch`, `--num_workers`

2. [scripts/train_text_cls.py](../scripts/train_text_cls.py)
- `--csv`, `--out`, `--test_size`, `--seed`

3. [scripts/bulk_ws_eval.py](../scripts/bulk_ws_eval.py)
- `--audio_dir`, `--csv_out`, `--url`, `--chunk_ms`, `--concurrency`, `--timeout_s`, `--patterns`, `--resume`, `--limit`

4. [client/ws_client.py](../client/ws_client.py)
- `--url`, `--wav`, `--chunk_ms`

## Interface Stability Notes
- `label` and `decision` are both present in the final response for compatibility.
- Runtime config mutates process globals and is therefore not connection-isolated.
- The sample client defaults to port `8080`, while the documented server run command uses `8989`.
- AGI and websocket protocols are convention-based; no formal schema versioning is committed.
