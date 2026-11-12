# 012 Deployment and Operations

## Build and Runtime Dependencies
From [requirements.txt](../requirements.txt):
- Inference: `faster-whisper`, `ctranslate2`, `torch`, `torchaudio`
- Service: `fastapi`, `uvicorn`, `websockets`
- Data/audio: `numpy`, `soundfile`, `librosa`, `pydub`
- ML: `scikit-learn`, `joblib`

Operational external dependencies:
- `ffmpeg` for audio conversion paths
- Elasticsearch when `ELASTIC_ENABLED=true`
- SFTP storage when `AUDIO_SAVE_ENABLED=true`
- Asterisk EAGI environment for [amd.py](../amd.py)
- Optional dashboard dependencies for [elastic_dashboard.py](../elastic_dashboard.py), such as `streamlit`, `plotly`, and `requests`

## Runbook: Server
Example commands verified in the repo:
- `uvicorn server.ws_server:app --host 0.0.0.0 --port 8989`
- `./run.sh --bg`

Required before start:
- Ensure `WHISPER_MODEL_DIR` and `CLASSIFIER_PATH` point to valid artifacts.
- Ensure device/compute settings match host hardware.
- Configure Elasticsearch and storage env values only if those features are needed.

## Runbook: Asterisk Client
- Deploy [amd.py](../amd.py) to the AGI directory.
- Set `AMD_WS_URL` to a reachable inference endpoint.
- Confirm FD3 audio availability and AGI permissions.

## Operational Checks
- `GET /health` before routing traffic
- `GET /status` for runtime confirmation
- `GET /elastic-status` when Elasticsearch logging is enabled
- `GET /audio-status` when audio saving is enabled

## Packaging and Infra State
Verified repository does not include:
- Dockerfile
- Kubernetes manifests
- committed systemd service units
- CI/CD workflows

The repository **does** include:
- [run.sh](../run.sh) for process management
- [.env](../.env) for deployment configuration
- [elastic_dashboard.py](../elastic_dashboard.py) for dashboarding

## Rollback Strategy (Operational Recommendation)
Because infra manifests are absent, practical rollback should be versioned artifact + process-level rollback:
1. Keep previous model and classifier artifacts available.
2. Maintain previous environment snapshot externally.
3. Repoint `WHISPER_MODEL_DIR` and `CLASSIFIER_PATH` to known-good versions.

## Operational Risks
- Startup may fail if model artifacts are missing or corrupt.
- No built-in TLS/auth controls exist in the service itself.
- Side-service failures (Elasticsearch or SFTP storage) do not halt inference, but they reduce auditability and observability.
- Import-time side effects make process health dependent on immediate dependency availability.
- The classifier may remain disabled if the trained artifact shape does not match runtime validation.
