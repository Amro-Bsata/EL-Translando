#!/usr/bin/env bash
# ============================================================
# run_server.sh - start FastAPI server + resource monitor on the Arduino
#
# Usage:  ./run_server.sh
# Stop:   Ctrl-C
# ============================================================

set -e

PYTHON=${PYTHON:-python3}
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./server_logs/$TS"
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
MON_LOG="$LOG_DIR/monitor.log"
MON_CSV="$LOG_DIR/resources.csv"

SERVER_PID=""
MON_PID=""

cleanup() {
    echo ""
    echo "[INFO] Shutting down..."
    [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[OK] Logs in: $LOG_DIR"
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting resource monitor..."
$PYTHON tools/resource_monitor.py --duration 0 --output "$MON_CSV" \
    > "$MON_LOG" 2>&1 &
MON_PID=$!
echo "[OK] Monitor PID: $MON_PID  (CSV: $MON_CSV)"

echo "[INFO] Starting FastAPI server..."
echo "[INFO] (server log: $SERVER_LOG)"
$PYTHON main.py 2>&1 | tee "$SERVER_LOG"
SERVER_PID=$!
