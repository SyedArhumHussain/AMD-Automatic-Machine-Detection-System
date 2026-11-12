# 002 Repository Structure

## Top-Level Layout
- [server](../server): main inference service implementation.
- [client](../client): sample WebSocket client for WAV streaming.
- [scripts](../scripts): training and evaluation tooling.
- [models](../models): runtime and training model artifacts.
- [data](../data): data assets and dataset-related files.
- [docs](../docs): forensic and operational documentation set.
- [amd.py](../amd.py): Asterisk EAGI bridge entrypoint.
- [elastic_logger.py](../elastic_logger.py): Elasticsearch bulk logger.
- [audio_logger.py](../audio_logger.py): SFTP audio uploader.
- [elastic_dashboard.py](../elastic_dashboard.py): Streamlit analytics dashboard.
- [run.sh](../run.sh): startup and process-management helper.
- [requirements.txt](../requirements.txt): Python dependency list.
- [README.md](../README.md): operator-facing entrypoint documentation.
- [.env](../.env): deployment configuration snapshot; keep it protected and out of source control.

## Directory-by-Directory Responsibilities
### `server/`
- Primary runtime service boundary.
- Contains the FastAPI app, websocket handler, preprocessing logic, VAD integration, ASR, classification logic, and monitoring endpoints.

### `client/`
- Contains [client/ws_client.py](../client/ws_client.py), a simple reference client for manual testing.

### `scripts/`
- [scripts/finetune_whisper_small.py](../scripts/finetune_whisper_small.py): Whisper fine-tuning pipeline.
- [scripts/train_text_cls.py](../scripts/train_text_cls.py): TF-IDF + Logistic Regression classifier training.
- [scripts/bulk_ws_eval.py](../scripts/bulk_ws_eval.py): concurrent websocket benchmarking and CSV export.
- [scripts/utils.py](../scripts/utils.py): helper functions and vocabulary assets for training/analysis paths.

### `models/`
- `whisper-ct2`: deployment-ready CTranslate2 Whisper assets.
- `whisper-small-finetuned`: Hugging Face style model output and checkpoint history.
- `text_cls_amd_v2.joblib`: serialized classifier bundle used by the server.
- `Train-Test-2.ipynb`: exploratory notebook artifact.

### `docs/`
- Ordered forensic, architectural, operational, and maintenance documentation following the `NNN-topic.md` convention.

## Naming Conventions
- Documentation files follow `NNN-topic.md` numbering for ordered reading.
- Scripts use verb-led names such as `train_`, `finetune_`, and `bulk_`.
- Model folders are artifact-type oriented (`whisper-ct2`, `whisper-small-finetuned`).
- Runtime code is organized by operational concern rather than by package hierarchy.

## Architectural Role Mapping
- Runtime hot path: [amd.py](../amd.py) <-> [server/ws_server.py](../server/ws_server.py)
- Persistence side path: [server/ws_server.py](../server/ws_server.py) -> [elastic_logger.py](../elastic_logger.py) and [audio_logger.py](../audio_logger.py)
- Model lifecycle path: [scripts](../scripts) -> [models](../models) -> [server/ws_server.py](../server/ws_server.py)
- Ops visibility path: [elastic_dashboard.py](../elastic_dashboard.py) -> Elasticsearch indices

## Structure Risks
- Large model artifacts are committed in-repo, increasing repository weight and clone time.
- The workspace includes deployment-specific `.env` content, which increases handling risk if not protected.
- Some operational assumptions live in scripts like [run.sh](../run.sh) and are path-specific to `/root/amd-server`.
