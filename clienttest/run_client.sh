#!/usr/bin/env bash
# ============================================================
# run_client.sh - run all client tests against a remote server
#
# Usage:
#   ./run_client.sh --url http://192.168.1.42:5000
#   ./run_client.sh --url http://192.168.1.42:5000 --only text
#   ./run_client.sh --url http://192.168.1.42:5000 --only stability
# ============================================================

set -e

PYTHON=${PYTHON:-python3}
URL=""
ONLY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --url)  URL="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$URL" ]; then
    echo "[ERR] --url is required (e.g. --url http://192.168.1.42:5000)"
    exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="./client_logs/$TS"
mkdir -p "$OUT"
echo "[INFO] Output directory: $OUT"
echo "[INFO] Server URL:       $URL"
echo ""

run_text() {
    echo "=== TEXT BENCHMARK (DE -> EN) ==="
    $PYTHON client_benchmark.py \
        --url "$URL" --direction de2en --repeats 3 \
        --corpus corpus_de.json \
        --output  "$OUT/text_de2en.csv" \
        --summary "$OUT/text_de2en_summary.txt"
    echo ""
    echo "=== TEXT BENCHMARK (EN -> DE) ==="
    $PYTHON client_benchmark.py \
        --url "$URL" --direction en2de --repeats 3 \
        --corpus corpus_en.json \
        --output  "$OUT/text_en2de.csv" \
        --summary "$OUT/text_en2de_summary.txt"
}

run_audio() {
    echo "=== AUDIO BENCHMARK (TTS-generated, DE -> EN) ==="
    $PYTHON client_audio_test.py \
        --url "$URL" --mode tts --direction de2en --repeats 2 \
        --corpus corpus_de.json \
        --output  "$OUT/audio_tts_de2en.csv" \
        --summary "$OUT/audio_tts_de2en_summary.txt"
}

run_stability() {
    echo "=== STABILITY TEST (500 mixed requests) ==="
    $PYTHON client_stability_test.py \
        --url "$URL" --mixed --requests 500 --delay 0.3 \
        --output  "$OUT/stability_results.csv" \
        --buckets "$OUT/stability_buckets.csv" \
        --summary "$OUT/stability_summary.txt"
}

case "$ONLY" in
    text)      run_text ;;
    audio)     run_audio ;;
    stability) run_stability ;;
    "")        run_text; echo ""; run_audio; echo ""; run_stability ;;
    *) echo "[ERR] Unknown --only value: $ONLY  (use: text|audio|stability)"; exit 1 ;;
esac

echo ""
echo "[OK] All done. Artifacts in: $OUT"
ls -la "$OUT"
