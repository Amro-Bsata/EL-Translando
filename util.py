"""
Utility-Modul für den Edge-Übersetzer.
Optimiert für Geräte mit begrenztem RAM (~4GB).

Verbesserungen gegenüber Original:
- faster-whisper (CTranslate2) statt openai-whisper → 4x schneller auf CPU
- Modelle werden EINMAL geladen und im RAM gehalten (Singleton-Pattern)
- Konsistente Return-Werte (immer str, nie Liste)
- Kein toter/auskommentierter Code
"""

import subprocess
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import config

# ============================================================
# Globale Modell-Instanzen (Singleton — einmal laden, immer nutzen)
# ============================================================
_whisper_model = None
_translator_de_en = None
_translator_en_de = None
_tokenizer_de_en = None
_tokenizer_en_de = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print(f"[INIT] Lade Whisper '{config.WHISPER_MODEL_SIZE}' "
              f"(device={config.WHISPER_DEVICE}, compute={config.WHISPER_COMPUTE_TYPE})...")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        print("[INIT] Whisper geladen.")
    return _whisper_model


def _get_translator_de_en():
    global _translator_de_en, _tokenizer_de_en
    if _translator_de_en is None:
        print(f"[INIT] Lade Übersetzungsmodell DE→EN: {config.TRANSLATION_MODEL_DE_EN}")
        _tokenizer_de_en = MarianTokenizer.from_pretrained(config.TRANSLATION_MODEL_DE_EN)
        _translator_de_en = MarianMTModel.from_pretrained(config.TRANSLATION_MODEL_DE_EN)
        print("[INIT] DE→EN geladen.")
    return _translator_de_en, _tokenizer_de_en


def _get_translator_en_de():
    global _translator_en_de, _tokenizer_en_de
    if _translator_en_de is None:
        print(f"[INIT] Lade Übersetzungsmodell EN→DE: {config.TRANSLATION_MODEL_EN_DE}")
        _tokenizer_en_de = MarianTokenizer.from_pretrained(config.TRANSLATION_MODEL_EN_DE)
        _translator_en_de = MarianMTModel.from_pretrained(config.TRANSLATION_MODEL_EN_DE)
        print("[INIT] EN→DE geladen.")
    return _translator_en_de, _tokenizer_en_de


# ============================================================
# Speech-to-Text
# ============================================================

def voice_to_text(audio_path: str) -> str:
    """Transkribiert eine Audiodatei zu Text (faster-whisper)."""
    model = _get_whisper()
    print(f"[STT] Transkribiere: {audio_path}")
    segments, info = model.transcribe(audio_path, beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    print(f"[STT] Ergebnis ({info.language}): {text}")
    return text


# ============================================================
# Übersetzung — konsistente Rückgabe: immer str
# ============================================================

def translate_de_to_en(text: str) -> str:
    """Deutsch → Englisch. Gibt einen String zurück."""
    model, tokenizer = _get_translator_de_en()
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = tokenizer.decode(model.generate(**inputs)[0], skip_special_tokens=True)
    print(f"[TRANSLATE] DE→EN: '{text}' → '{translated}'")
    return translated


def translate_en_to_de(text: str) -> str:
    """Englisch → Deutsch. Gibt einen String zurück."""
    model, tokenizer = _get_translator_en_de()
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = tokenizer.decode(model.generate(**inputs)[0], skip_special_tokens=True)
    print(f"[TRANSLATE] EN→DE: '{text}' → '{translated}'")
    return translated


# ============================================================
# Text-to-Speech (Piper)
# ============================================================

def text_to_voice(text: str, model_path: str, output_path: str) -> bool:
    """Erzeugt eine WAV-Datei via Piper TTS."""
    # Sicherheit: Anführungszeichen escapen
    safe_text = text.replace('"', '\\"')
    cmd = f'echo "{safe_text}" | {config.PIPER_BINARY} -m {model_path} -f {output_path}'
    print(f"[TTS] Kommando: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[TTS] Gespeichert: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[TTS] FEHLER: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[TTS] FEHLER: 'piper' nicht im PATH gefunden.")
        return False


# ============================================================
# Stereo-Kanal-Verarbeitung
# ============================================================

def process_to_stereo_channels(input_file: str):
    """
    Liest Audio → erzeugt Links-nur und Rechts-nur Stereo-Signale.
    Returns: (signal_left, signal_right, samplerate) oder (None, None, None)
    """
    try:
        data, fs = sf.read(input_file, dtype="float32")
        print(f"[STEREO] Geladen: {input_file} | SR={fs} | Shape={data.shape}")

        mono = data.mean(axis=1) if data.ndim == 2 else data
        n = len(mono)

        signal_left = np.zeros((n, 2), dtype="float32")
        signal_left[:, 0] = mono

        signal_right = np.zeros((n, 2), dtype="float32")
        signal_right[:, 1] = mono

        return signal_left, signal_right, fs

    except Exception as e:
        print(f"[STEREO] FEHLER: {e}")
        return None, None, None


# ============================================================
# Modelle beim Import vorladen (optional, für schnelleren 1. Request)
# ============================================================

def preload_models():
    """Alle Modelle vorab laden. Aufrufen beim Server-Start."""
    print("=" * 50)
    print("Lade alle Modelle vor...")
    _get_whisper()
    _get_translator_de_en()
    _get_translator_en_de()  # noqa: side-effect is caching
    print("Alle Modelle geladen!")
    print("=" * 50)