"""
Konfiguration für den Edge-Übersetzer.
Angepasst für Geräte mit ~4GB RAM (z.B. Raspberry Pi 4).
"""
import os

# --- Server ---
PORT = 5000
HOST = "0.0.0.0"
DEBUG = False
# --- Ordner ---Du bist wundertoll! --- (Uploads, Outputs, Static Content)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
STATIC_FOLDER = "static_content"

# --- Ngrok (optional, aus Umgebungsvariable laden!) ---
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "2wBKuYE4bKdMKyArayVrYKEuldK_29suq6Fij4pnp1BJBkSpr")
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "noncollapsible-garfield-odorously.ngrok-free.dev")

# --- Whisper (STT) ---
# "tiny" = ~39MB VRAM, schnellster; "base" = ~74MB, guter Kompromiss
# "small" = ~244MB — nur wenn genug RAM frei ist
WHISPER_MODEL_SIZE = "tiny"
# Device: "cpu" für Pi/Arduino-ähnliche Boards, "cuda" falls GPU vorhanden
WHISPER_DEVICE = "cpu"
# CTranslate2 Quantisierung: "int8" spart ~50% RAM vs float32
WHISPER_COMPUTE_TYPE = "int8"

# --- Übersetzung (Helsinki-NLP via CTranslate2) ---
TRANSLATION_MODEL_DE_EN = "Helsinki-NLP/opus-mt-de-en"
TRANSLATION_MODEL_EN_DE = "Helsinki-NLP/opus-mt-en-de"

# --- TTS (Piper — bereits edge-optimiert) ---
PIPER_BINARY = "piper"
PIPER_MODEL_EN = "en_GB-jenny_dioco-medium.onnx"
PIPER_MODEL_DE = "de_DE-thorsten-medium.onnx"

# --- Ordner erstellen ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
