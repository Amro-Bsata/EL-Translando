"""
Edge-Übersetzer — Flask Server
Optimiert für Geräte mit ~4GB RAM.
"""

import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import soundfile as sf

import config
import util

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__, static_folder=config.STATIC_FOLDER, static_url_path="/static")
CORS(app)


# --- Seiten ---

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "website.html")


# --- Audio-Upload Routen ---

@app.route("/upload_de", methods=["POST"])
def upload_german_audio():
    """DE-Audio → STT → Translate EN → TTS → Stereo Links"""
    return _handle_audio_upload(
        lang="de",
        translate_fn=util.translate_de_to_en,
        channel="left",
        tts_model=config.PIPER_MODEL_EN,
    )


@app.route("/upload_en", methods=["POST"])
def upload_english_audio():
    """EN-Audio → STT → Translate DE → TTS → Stereo Rechts"""
    return _handle_audio_upload(
        lang="en",
        translate_fn=util.translate_en_to_de,
        channel="right",
        tts_model=config.PIPER_MODEL_DE,
    )


# --- Text-Upload Routen ---

@app.route("/upload_text_de", methods=["POST"])
def upload_german_text():
    """DE-Text → Translate EN → TTS → Stereo Links"""
    return _handle_text_upload(
        translate_fn=util.translate_de_to_en,
        channel="left",
        tts_model=config.PIPER_MODEL_EN,
    )


@app.route("/upload_text_en", methods=["POST"])
def upload_english_text():
    """EN-Text → Translate DE → TTS → Stereo Rechts"""
    return _handle_text_upload(
        translate_fn=util.translate_en_to_de,
        channel="right",
        tts_model=config.PIPER_MODEL_DE,
    )


# --- Audio-Dateien ausliefern ---

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUT_FOLDER, filename)


# ============================================================
# Interne Helfer (DRY — kein duplizierter Code mehr)
# ============================================================

def _handle_audio_upload(lang, translate_fn, channel, tts_model):
    """Gemeinsame Logik für Audio-Uploads."""
    if "audio_data" not in request.files:
        return jsonify({"error": "Keine Audiodaten"}), 400

    ts = str(int(time.time()))
    input_path = os.path.join(config.UPLOAD_FOLDER, f"input_{lang}_{ts}.wav")

    try:
        request.files["audio_data"].save(input_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    # Pipeline: STT → Translate → TTS → Stereo
    text = util.voice_to_text(input_path)
    if not text:
        return jsonify({"status": "error", "message": "STT fehlgeschlagen"}), 500

    translated = translate_fn(text)
    audio_url = _text_to_stereo_audio(translated, ts, channel, tts_model)

    # Aufräumen
    _safe_remove(input_path)

    if not audio_url:
        return jsonify({"status": "error", "message": "Audio-Erzeugung fehlgeschlagen"}), 500

    return jsonify({"status": "success", "audio_url": audio_url, "original": text, "translated": translated})


def _handle_text_upload(translate_fn, channel, tts_model):
    """Gemeinsame Logik für Text-Uploads."""
    data = request.get_json(silent=True)
    text = data.get("text", "").strip() if data else ""
    if not text:
        return jsonify({"error": "Kein Text"}), 400

    ts = str(int(time.time()))
    translated = translate_fn(text)
    audio_url = _text_to_stereo_audio(translated, ts, channel, tts_model)

    if not audio_url:
        return jsonify({"status": "error", "message": "Audio-Erzeugung fehlgeschlagen"}), 500

    return jsonify({"status": "success", "audio_url": audio_url, "translated": translated})


def _text_to_stereo_audio(text, ts, channel, model_path):
    """TTS → Stereo-Kanal → WAV speichern → URL zurückgeben."""
    temp_path = os.path.join(config.OUTPUT_FOLDER, f"temp_{ts}.wav")

    if not util.text_to_voice(text, model_path, temp_path):
        return None

    signal_left, signal_right, fs = util.process_to_stereo_channels(temp_path)
    if signal_left is None:
        return None

    signal = signal_left if channel == "left" else signal_right
    output_name = f"output_{channel}_{ts}.wav"
    output_path = os.path.join(config.OUTPUT_FOLDER, output_name)

    try:
        sf.write(output_path, signal, fs)
    except Exception as e:
        print(f"[ERROR] Speichern fehlgeschlagen: {e}")
        return None
    finally:
        _safe_remove(temp_path)

    return f"/outputs/{output_name}"


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    # Modelle beim Start laden (nicht beim ersten Request)
    util.preload_models()

    # Ngrok (optional)
    if config.NGROK_AUTH_TOKEN and config.NGROK_DOMAIN:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
            tunnel = ngrok.connect(config.PORT, domain=config.NGROK_DOMAIN)
            print(f"[NGROK] Tunnel: {tunnel.public_url}")
        except Exception as e:
            print(f"[NGROK] Fehler: {e}")
    else:
        print("[INFO] Kein Ngrok konfiguriert. Läuft nur lokal.")

    print(f"[SERVER] http://{config.HOST}:{config.PORT}")
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
