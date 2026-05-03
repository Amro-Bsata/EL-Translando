"""
El-Translando Server - FastAPI version.
Runs on the Arduino UNO Q. Exposes translation endpoints with
per-stage timing in every response.

NEW in this revision:
    - per-request TTS engine selection ("tts_engine": "pyttsx3" | "piper")
    - serves a minimal web frontend at GET / and /ui

Endpoints:
    GET  /                        - web frontend (frontend.html)
    GET  /ui                      - same as /
    GET  /health                  - liveness probe
    GET  /info                    - model + config metadata
    POST /translate/text          - body: {text, direction, tts_engine?}
    POST /translate/audio         - multipart: audio_file, direction, tts_engine?
    GET  /outputs/{file}          - serves generated WAVs
    GET  /docs                    - auto-generated OpenAPI UI

Run:
    pip install -r requirements_server.txt
    python3 main.py
        OR
    uvicorn main:app --host 0.0.0.0 --port 5000
"""

import os
import time
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import soundfile as sf

import config
import util


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="El-Translando",
    description="Offline edge translation pipeline (STT -> MT -> TTS)",
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic models
# ============================================================
TtsEngineLiteral = Literal["pyttsx3", "piper"]


class TextTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    direction: Literal["de2en", "en2de"]
    tts_engine: Optional[TtsEngineLiteral] = "pyttsx3"


class Timings(BaseModel):
    stt_ms: Optional[float] = None
    mt_ms: Optional[float] = None
    tts_ms: Optional[float] = None
    total_ms: float
    audio_duration_ms: Optional[float] = None


class TranslateResponse(BaseModel):
    status: str
    original: Optional[str] = None
    translated: str
    audio_url: Optional[str] = None
    tts_engine: Optional[str] = None
    timings: Timings


# ============================================================
# Frontend
# ============================================================
_FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.html")


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def serve_frontend():
    if not os.path.exists(_FRONTEND_PATH):
        return HTMLResponse("<h3>frontend.html not found next to main.py</h3>",
                            status_code=404)
    with open(_FRONTEND_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ============================================================
# Health and metadata
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/info")
def info():
    return {
        "whisper_model": config.WHISPER_MODEL_SIZE,
        "whisper_device": config.WHISPER_DEVICE,
        "whisper_compute": config.WHISPER_COMPUTE_TYPE,
        "mt_model_de_en": config.TRANSLATION_MODEL_DE_EN,
        "mt_model_en_de": config.TRANSLATION_MODEL_EN_DE,
        "tts_engines_available": list(util.VALID_TTS_ENGINES),
        "tts_default": "pyttsx3",
        "tts_rate": config.TTS_RATE,
    }


# ============================================================
# Text endpoint
# ============================================================
@app.post("/translate/text", response_model=TranslateResponse)
def translate_text(req: TextTranslateRequest):
    t_total = time.perf_counter()

    if req.direction == "de2en":
        translate_fn = util.translate_de_to_en
        voice = config.TTS_VOICE_EN
        channel = "left"
        out_lang = "en"
    else:
        translate_fn = util.translate_en_to_de
        voice = config.TTS_VOICE_DE
        channel = "right"
        out_lang = "de"

    # MT
    t0 = time.perf_counter()
    translated = translate_fn(req.text)
    mt_ms = round((time.perf_counter() - t0) * 1000, 2)

    # TTS
    ts = str(int(time.time() * 1000))
    t0 = time.perf_counter()
    audio_url, audio_duration_ms = _tts_to_stereo(
        translated, ts, channel, voice,
        tts_engine=req.tts_engine or "pyttsx3",
        lang=out_lang,
    )
    tts_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not audio_url:
        raise HTTPException(status_code=500, detail="TTS generation failed")

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)

    return TranslateResponse(
        status="success",
        original=req.text,
        translated=translated,
        audio_url=audio_url,
        tts_engine=req.tts_engine or "pyttsx3",
        timings=Timings(
            mt_ms=mt_ms, tts_ms=tts_ms, total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
        ),
    )


# ============================================================
# Audio endpoint
# ============================================================
@app.post("/translate/audio", response_model=TranslateResponse)
async def translate_audio(
    audio_file: UploadFile = File(...),
    direction: Literal["de2en", "en2de"] = Form(...),
    tts_engine: Optional[str] = Form("pyttsx3"),
):
    t_total = time.perf_counter()

    # validate engine
    engine = (tts_engine or "pyttsx3").lower()
    if engine not in util.VALID_TTS_ENGINES:
        raise HTTPException(
            status_code=400,
            detail="tts_engine must be one of " + str(list(util.VALID_TTS_ENGINES)),
        )

    if direction == "de2en":
        translate_fn = util.translate_de_to_en
        voice = config.TTS_VOICE_EN
        channel = "left"
        in_lang = "de"
        out_lang = "en"
    else:
        translate_fn = util.translate_en_to_de
        voice = config.TTS_VOICE_DE
        channel = "right"
        in_lang = "en"
        out_lang = "de"

    ts = str(int(time.time() * 1000))
    input_path = os.path.join(config.UPLOAD_FOLDER, "input_" + in_lang + "_" + ts + ".wav")

    # Save upload
    try:
        contents = await audio_file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Upload save failed: " + str(e))

    # STT
    t0 = time.perf_counter()
    text = util.voice_to_text(input_path)
    stt_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not text:
        _safe_remove(input_path)
        raise HTTPException(status_code=500, detail="STT produced empty text")

    # MT
    t0 = time.perf_counter()
    translated = translate_fn(text)
    mt_ms = round((time.perf_counter() - t0) * 1000, 2)

    # TTS
    t0 = time.perf_counter()
    audio_url, audio_duration_ms = _tts_to_stereo(
        translated, ts, channel, voice,
        tts_engine=engine, lang=out_lang,
    )
    tts_ms = round((time.perf_counter() - t0) * 1000, 2)

    _safe_remove(input_path)

    if not audio_url:
        raise HTTPException(status_code=500, detail="TTS generation failed")

    total_ms = round((time.perf_counter() - t_total) * 1000, 2)

    return TranslateResponse(
        status="success",
        original=text,
        translated=translated,
        audio_url=audio_url,
        tts_engine=engine,
        timings=Timings(
            stt_ms=stt_ms, mt_ms=mt_ms, tts_ms=tts_ms, total_ms=total_ms,
            audio_duration_ms=audio_duration_ms,
        ),
    )


# ============================================================
# Serve generated audio
# ============================================================
@app.get("/outputs/{filename}")
def serve_output(filename: str):
    full = os.path.join(config.OUTPUT_FOLDER, filename)
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(full, media_type="audio/wav")


# ============================================================
# Internal helpers
# ============================================================
def _tts_to_stereo(text, ts, channel, voice, tts_engine="pyttsx3", lang=None):
    temp_path = os.path.join(config.OUTPUT_FOLDER, "temp_" + ts + ".wav")
    if not util.text_to_voice(text, voice, temp_path,
                              tts_engine=tts_engine, lang=lang):
        return None, 0
    try:
        sfinfo = sf.info(temp_path)
        audio_duration_ms = round(sfinfo.duration * 1000, 2)
    except Exception:
        audio_duration_ms = 0

    signal_left, signal_right, fs = util.process_to_stereo_channels(temp_path)
    if signal_left is None:
        return None, 0

    signal = signal_left if channel == "left" else signal_right
    output_name = "output_" + channel + "_" + ts + ".wav"
    output_path = os.path.join(config.OUTPUT_FOLDER, output_name)
    try:
        sf.write(output_path, signal, fs)
    except Exception as e:
        print("[ERROR] Saving failed: " + str(e))
        return None, 0
    finally:
        _safe_remove(temp_path)

    return "/outputs/" + output_name, audio_duration_ms


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ============================================================
# Startup: preload models
# ============================================================
@app.on_event("startup")
def on_startup():
    print("[STARTUP] Preloading models...")
    util.preload_models()
    print("[STARTUP] Ready.")


# ============================================================
# Run via `python main.py`
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
