#!/usr/bin/env bash
# -----------------------------------------------------------------------
#  setenv.sh – environment for ws_server.py
#  Source this BEFORE running the server:
#      source setenv.sh
#      python ws_server.py
# -----------------------------------------------------------------------

# ========================
#  Elasticsearch Logging
# ========================
export ELASTIC_ENABLED=true                  # set to false if you want to disable ES completely
export ELASTICSEARCH_URL="http://es.softotele.com:80"
export ELASTIC_API_KEY="NjFELUJwMEJyWmNsVjhTRmZZWDg6Y0RPZkwyZmNUcDI5aE93bS1BZFdtUQ=="
export ELASTIC_INDEX_PREFIX="amd-stats"      # daily indices will be amd-stats-YYYY.MM.DD

# Tuning for heavy loads (keep the defaults unless you know what you're doing)
export ELASTIC_BULK_SIZE="100"               # documents per bulk request
export ELASTIC_FLUSH_INTERVAL="2.0"          # seconds between flushes
export ELASTIC_MAX_QUEUE="50000"            # in‑memory queue size
export ELASTIC_NUM_WORKERS="3"              # number of background sender threads
export ELASTIC_BULK_TIMEOUT="30"            # seconds per bulk request
export ELASTIC_MAX_RETRIES="5"              # per‑document retry attempts
export ELASTIC_DEAD_LETTER="/var/log/amd_elastic_deadletter.jsonl"  # unrecoverable docs land here
export ELASTIC_QUEUE_WARN_PCT="0.7"         # warn when queue is 70% full

# Debug (set to true if you need very verbose Elasticsearch logging)
export ELASTIC_DEBUG=false

# ========================
#  Audio Storage (SFTP)
# ========================
export AUDIO_SAVE_ENABLED=true              # set to false if you don't want audio saved
export AUDIO_FORMAT="mp3"                   # format to save audio (mp3)
export AUDIO_SAVE_FILTER="all"              # "all", "human", "machine" – which calls to save

# Storage box credentials (matches your elastic_logger default)
export AUDIO_URL_USER="u563163"
export AUDIO_URL_PASSWORD='AMD$server!1122@ARC'
export AUDIO_URL_HOST="u563163.your-storagebox.de"
export AUDIO_URL_BASE_DIR="audio_storage"

# ========================
#  (Optional) WS Server tuning
# ========================
# export MAX_CALL_SEC=2.5
# export SILENCE_RMS_THRESHOLD=0.003
# export WHISPER_MODEL_DIR="small.en"
# export DEVICE="cuda"
# ... etc – check ws_server.py top section for all adjustable vars

# ========================
#  Final reminder
# ========================
echo "Environment set. You can now run:  python ws_server.py"