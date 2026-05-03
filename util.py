"""
Utility module for the Edge Translator.
Optimized for devices with limited RAM (~4GB).

Improvements:
- faster-whisper (CTranslate2) instead of openai-whisper -> 4x faster on CPU
- Models are loaded ONCE and kept in RAM (singleton pattern)
- Consistent return values (always str, never list)
- TWO TTS engines available: pyttsx3 (eSpeak-NG, very fast)
                         and Piper  (neural, higher quality)
  Selectable per request via the API parameter "tts_engine".
- No dead/commented code
"""

import os
import shutil
import subprocess
import threading
import numpy as np
import soundfile as sf
import pyttsx3
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import config

# ============================================================
# Global model instances (singleton -- load once, use always)
# ============================================================
_whisper_model = None
_translator_de_en = None
_translator_en_de = None
_tokenizer_de_en = None
_tokenizer_en_de = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[INIT] Loading Whisper '" + config.WHISPER_MODEL_SIZE + "' "
              "(device=" + config.WHISPER_DEVICE +
              ", compute=" + config.WHISPER_COMPUTE_TYPE + ")...")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        print("[INIT] Whisper loaded.")
    return _whisper_model


def _get_translator_de_en():
    global _translator_de_en, _tokenizer_de_en
    if _translator_de_en is None:
        print("[INIT] Loading translation model DE-EN: " + config.TRANSLATION_MODEL_DE_EN)
        _tokenizer_de_en = MarianTokenizer.from_pretrained(config.TRANSLATION_MODEL_DE_EN)
        _translator_de_en = MarianMTModel.from_pretrained(config.TRANSLATION_MODEL_DE_EN)
        print("[INIT] DE-EN loaded.")
    return _translator_de_en, _tokenizer_de_en


def _get_translator_en_de():
    global _translator_en_de, _tokenizer_en_de
    if _translator_en_de is None:
        print("[INIT] Loading translation model EN-DE: " + config.TRANSLATION_MODEL_EN_DE)
        _tokenizer_en_de = MarianTokenizer.from_pretrained(config.TRANSLATION_MODEL_EN_DE)
        _translator_en_de = MarianMTModel.from_pretrained(config.TRANSLATION_MODEL_EN_DE)
        print("[INIT] EN-DE loaded.")
    return _translator_en_de, _tokenizer_en_de


# ============================================================
# Speech-to-Text
# ============================================================

def voice_to_text(audio_path):
    """Transcribes an audio file to text (faster-whisper)."""
    model = _get_whisper()
    print("[STT] Transcribing: " + audio_path)
    segments, info = model.transcribe(audio_path, beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    print("[STT] Result (" + info.language + "): " + text)
    return text


# ============================================================
# Translation -- consistent return: always str
# ============================================================

def translate_de_to_en(text):
    """German to English. Returns a string."""
    model, tokenizer = _get_translator_de_en()
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = tokenizer.decode(model.generate(**inputs)[0], skip_special_tokens=True)
    print("[TRANSLATE] DE->EN: '" + text + "' -> '" + translated + "'")
    return translated


def translate_en_to_de(text):
    """English to German. Returns a string."""
    model, tokenizer = _get_translator_en_de()
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = tokenizer.decode(model.generate(**inputs)[0], skip_special_tokens=True)
    print("[TRANSLATE] EN->DE: '" + text + "' -> '" + translated + "'")
    return translated


# ============================================================
# Text-to-Speech: pyttsx3 (eSpeak-NG)
# ============================================================

# Lock: pyttsx3 is NOT thread-safe, but FastAPI/uvicorn serves
# parallel requests.
_tts_lock = threading.Lock()


def _select_voice(engine, voice_identifier):
    """Selects a pyttsx3 voice via substring match. Returns True if found."""
    target = voice_identifier.lower()
    for v in engine.getProperty("voices"):
        haystack = (str(v.id) + " " + str(v.name) + " " + str(getattr(v, "languages", ""))).lower()
        if target in haystack:
            engine.setProperty("voice", v.id)
            return True
    return False


def _tts_pyttsx3(text, voice_identifier, output_path):
    """pyttsx3 / eSpeak-NG synthesis. Returns True on success."""
    with _tts_lock:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_RATE)
            engine.setProperty("volume", config.TTS_VOLUME)

            if not _select_voice(engine, voice_identifier):
                print("[TTS-pyttsx3] WARN: No voice for '" + voice_identifier + "' -- using default.")

            engine.save_to_file(text, output_path)
            engine.runAndWait()
            engine.stop()
            del engine
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            print("[TTS-pyttsx3] ERROR: " + str(e))
            return False


# ============================================================
# Text-to-Speech: Piper (neural, ONNX)
# ============================================================

def _piper_model_for_lang(lang):
    """Returns the path to the Piper ONNX model for a language."""
    if lang == "de":
        return getattr(config, "PIPER_MODEL_DE", None)
    if lang == "en":
        return getattr(config, "PIPER_MODEL_EN", None)
    return None


def _tts_piper(text, lang, output_path):
    """
    Piper synthesis via the piper binary.
    Requires:
      - the 'piper' binary on PATH (or set config.PIPER_BINARY)
      - ONNX models at config.PIPER_MODEL_DE / PIPER_MODEL_EN
    Returns True on success.
    """
    binary = getattr(config, "PIPER_BINARY", "piper")
    if shutil.which(binary) is None:
        print("[TTS-piper] ERROR: '" + binary + "' not found on PATH.")
        return False

    model_path = _piper_model_for_lang(lang)
    if not model_path or not os.path.exists(model_path):
        print("[TTS-piper] ERROR: Piper model missing for lang=" + str(lang)
              + " (looked for: " + str(model_path) + ")")
        return False

    try:
        # piper reads text on stdin, writes WAV to -f
        proc = subprocess.run(
            [binary, "-m", model_path, "-f", output_path],
            input=text,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0:
            print("[TTS-piper] ERROR: rc=" + str(proc.returncode)
                  + " stderr=" + (proc.stderr or "").strip())
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print("[TTS-piper] ERROR: " + str(e))
        return False


# ============================================================
# Public TTS dispatcher -- chooses engine per request
# ============================================================

VALID_TTS_ENGINES = ("pyttsx3", "piper")


def text_to_voice(text, voice_identifier, output_path, tts_engine="pyttsx3", lang=None):
    """
    Creates a WAV file via the selected TTS engine.

    Args:
        text:              Text to synthesize.
        voice_identifier:  Substring to match the voice (pyttsx3 only,
                           e.g. "english", "de").
        output_path:       Target WAV path.
        tts_engine:        "pyttsx3" (default, fast) or "piper" (neural).
        lang:              Output language code ("de"/"en"), needed by Piper
                           to pick the correct ONNX model.

    Returns True if a non-empty WAV was produced.
    """
    if not text or not text.strip():
        print("[TTS] ERROR: Empty text.")
        return False

    engine_name = (tts_engine or "pyttsx3").lower()
    if engine_name not in VALID_TTS_ENGINES:
        print("[TTS] ERROR: Unknown engine '" + str(tts_engine)
              + "' -- must be one of " + str(VALID_TTS_ENGINES))
        return False

    if engine_name == "piper":
        # If no lang given, try to derive from voice_identifier
        if lang is None:
            v = (voice_identifier or "").lower()
            if "de" in v or "german" in v:
                lang = "de"
            elif "en" in v or "english" in v:
                lang = "en"
        ok = _tts_piper(text, lang or "en", output_path)
    else:
        ok = _tts_pyttsx3(text, voice_identifier, output_path)

    if ok:
        print("[TTS-" + engine_name + "] Saved: " + output_path)
    else:
        print("[TTS-" + engine_name + "] FAILED for: " + output_path)
    return ok


# ============================================================
# Stereo channel processing
# ============================================================

def process_to_stereo_channels(input_file):
    """
    Reads audio -> produces left-only and right-only stereo signals.
    Returns: (signal_left, signal_right, samplerate) or (None, None, None)
    """
    try:
        data, fs = sf.read(input_file, dtype="float32")
        print("[STEREO] Loaded: " + input_file + " | SR=" + str(fs) + " | Shape=" + str(data.shape))

        mono = data.mean(axis=1) if data.ndim == 2 else data
        n = len(mono)

        signal_left = np.zeros((n, 2), dtype="float32")
        signal_left[:, 0] = mono

        signal_right = np.zeros((n, 2), dtype="float32")
        signal_right[:, 1] = mono

        return signal_left, signal_right, fs

    except Exception as e:
        print("[STEREO] ERROR: " + str(e))
        return None, None, None


# ============================================================
# Preload models at import (optional, for faster first request)
# ============================================================

def preload_models():
    """Load all models upfront. Call at server startup."""
    print("=" * 50)
    print("Loading all models...")
    _get_whisper()
    _get_translator_de_en()
    _get_translator_en_de()
    print("All models loaded!")
    print("=" * 50)
