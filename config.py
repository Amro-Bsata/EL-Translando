"""
Konfiguration für den Edge-Übersetzer.
Angepasst für Geräte mit ~4GB RAM (z.B. Raspberry Pi 4).
"""
import os

# --- Server ---
PORT = 5000
HOST = "0.0.0.0"
DEBUG = False

# --- Ordner ---
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# --- Ngrok (optional) ---
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "")

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

# --- TTS (pyttsx3 — nutzt OS-native Engines: eSpeak-NG auf Linux) ---
# Vorteil: ~10-50x schneller als Piper. Nachteil: robotischer Klang.
# Rate = Wörter pro Minute (Default 200). 150-180 wirkt natürlicher.
TTS_RATE = 170
# Lautstärke (0.0 – 1.0)
TTS_VOLUME = 1.0
# Voice-Identifier: Substring, gegen den voice.id bzw. voice.name gematcht wird.
# Auf Linux/eSpeak sind typische IDs "english", "german" oder Locale-Codes "en", "de".
TTS_VOICE_EN = "english"
TTS_VOICE_DE = "german"



PIPER_MODEL_DE = "models/de_DE-thorsten-low.onnx"
PIPER_MODEL_EN = "models/en_US-amy-low.onnx"

# --- Ordner erstellen ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
