#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  AMD Whisper WS Server — Startup Script                                 ║
# ║  Usage: ./run.sh              (foreground)                               ║
# ║         ./run.sh --bg         (background with nohup)                    ║
# ║         ./run.sh --stop       (kill running server)                      ║
# ║         ./run.sh --status     (check health)                             ║
# ║         ./run.sh --logs       (tail journal logs)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR="/root/amd-server"
SERVER_DIR="${PROJECT_DIR}/server"
VENV_DIR="${PROJECT_DIR}/venv"
ENV_FILE="${PROJECT_DIR}/.env"
PID_FILE="${PROJECT_DIR}/amd-server.pid"
LOG_FILE="${PROJECT_DIR}/logs/amd-server.log"

HOST="0.0.0.0"
PORT="8989"

# ── Colors ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[✗]${NC} $1"; }
header(){ echo -e "\n${CYAN}═══ $1 ═══${NC}"; }

# ── Load .env ────────────────────────────────────────────────────────────
load_env() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
        info "Loaded .env"
    else
        warn "No .env found at $ENV_FILE — using defaults"
    fi
}

# ── Pre-flight checks ───────────────────────────────────────────────────
preflight() {
    header "Pre-flight Checks"

    # Virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        err "Virtual environment not found at $VENV_DIR"
        echo "    Run: python3 -m venv $VENV_DIR"
        exit 1
    fi
    info "Virtual environment: OK"

    # Activate venv
    source "${VENV_DIR}/bin/activate"
    info "Activated venv ($(python3 --version))"

    # Check key packages
    python3 -c "import faster_whisper" 2>/dev/null && info "faster-whisper: OK" || err "faster-whisper: MISSING"
    python3 -c "import torch" 2>/dev/null && info "torch: OK" || err "torch: MISSING"
    python3 -c "import fastapi" 2>/dev/null && info "fastapi: OK" || err "fastapi: MISSING"
    python3 -c "import soxr" 2>/dev/null && info "soxr: OK" || err "soxr: MISSING"

    # GPU check
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        info "CUDA GPU: ${GPU_NAME}"
    else
        warn "No CUDA GPU detected — using CPU mode"
        export DEVICE=cpu
        export COMPUTE_TYPE=int8
    fi

    # Classifier
    CLASSIFIER="${PROJECT_DIR}/models/text_cls_amd_v2.joblib"
    if [ -f "$CLASSIFIER" ]; then
        info "ML Classifier: OK ($(du -h "$CLASSIFIER" | cut -f1))"
    else
        warn "ML Classifier not found at $CLASSIFIER — keyword-only mode"
    fi

    # Server directory
    if [ ! -f "${SERVER_DIR}/ws_server.py" ]; then
        err "ws_server.py not found in $SERVER_DIR"
        exit 1
    fi
    info "ws_server.py: OK"

    # Log directory
    mkdir -p "${PROJECT_DIR}/logs"
    info "Log directory: OK"
}

# ── Check if running ────────────────────────────────────────────────────
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    # Also check by port
    if lsof -i ":${PORT}" -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ── Commands ─────────────────────────────────────────────────────────────
cmd_start() {
    load_env
    preflight

    if is_running; then
        warn "Server already running on port $PORT"
        cmd_status
        exit 0
    fi

    header "Starting AMD Server"
    cd "$SERVER_DIR"

    echo -e "  Host: ${HOST}:${PORT}"
    echo -e "  Device: ${DEVICE:-cuda}"
    echo -e "  Model: ${WHISPER_MODEL_DIR:-small.en}"
    echo ""

    exec uvicorn ws_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers 1 \
        --log-level warning \
        --timeout-keep-alive 30 \
        --ws-max-size 16777216
}

cmd_start_bg() {
    load_env
    preflight

    if is_running; then
        warn "Server already running on port $PORT"
        cmd_status
        exit 0
    fi

    header "Starting AMD Server (background)"
    cd "$SERVER_DIR"

    nohup "${VENV_DIR}/bin/uvicorn" ws_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers 1 \
        --log-level warning \
        --timeout-keep-alive 30 \
        --ws-max-size 16777216 \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    info "Started with PID $(cat $PID_FILE)"
    info "Logs: tail -f $LOG_FILE"

    # Wait for health
    echo -n "  Waiting for server "
    for i in $(seq 1 30); do
        sleep 2
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo ""
            info "Server is HEALTHY"
            cmd_status
            return 0
        fi
        echo -n "."
    done
    echo ""
    warn "Server not responding yet — check: tail -f $LOG_FILE"
}

cmd_stop() {
    header "Stopping AMD Server"

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            info "Sent SIGTERM to PID $PID"
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
                warn "Force killed PID $PID"
            fi
        fi
        rm -f "$PID_FILE"
    fi

    # Kill by port as fallback
    PIDS=$(lsof -i ":${PORT}" -t 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        info "Killed processes on port $PORT"
    fi

    info "Server stopped"
}

cmd_status() {
    header "Server Status"

    HEALTH=$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null) || {
        err "Server not responding on port $PORT"
        return 1
    }

    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
    info "Server is running"
}

cmd_logs() {
    if systemctl is-active amd-server >/dev/null 2>&1; then
        journalctl -u amd-server -f --no-pager
    elif [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        err "No logs found"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────
case "${1:-}" in
    --bg|--background)  cmd_start_bg ;;
    --stop)             cmd_stop ;;
    --status)           cmd_status ;;
    --logs)             cmd_logs ;;
    --help|-h)
        echo "Usage: $0 [--bg|--stop|--status|--logs|--help]"
        echo ""
        echo "  (no args)   Start in foreground"
        echo "  --bg        Start in background with nohup"
        echo "  --stop      Stop running server"
        echo "  --status    Check server health"
        echo "  --logs      Tail server logs"
        ;;
    *)                  cmd_start ;;
esac