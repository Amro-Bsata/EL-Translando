"""
tts_compare.py

Directly benchmarks Piper vs pyttsx3 on identical sentences.
Must be run on the UNO Q itself (both engines need to be installed).

For each sentence and each engine, measures:
    - synthesis time (ms)
    - produced audio duration (ms)
    - real-time factor RTF = synth_time / audio_duration
    - output file size (bytes)

Output: tts_compare.csv with one row per (sentence, engine) combination.

Usage:
    python3 tts_compare.py --engine both
    python3 tts_compare.py --engine pyttsx3 --lang en
    python3 tts_compare.py --engine piper --piper-model /path/to/model.onnx
"""

import argparse
import csv
import os
import subprocess
import time
import tempfile


SENTENCES_DE = [
    "Hallo, wie geht es dir?",
    "Wo ist der naechste Bahnhof?",
    "Ich haette gerne eine Tasse Kaffee.",
    "Entschuldigung, koennen Sie mir helfen?",
    "Das Wetter ist heute sehr schoen.",
    "Wie viel kostet das?",
    "Ich verstehe kein Deutsch.",
    "Wo befindet sich die Toilette?",
    "Koennten Sie bitte langsamer sprechen?",
    "Vielen Dank fuer Ihre Hilfe, das war wirklich sehr nett von Ihnen.",
]

SENTENCES_EN = [
    "Hello, how are you?",
    "Where is the nearest train station?",
    "I would like a cup of coffee.",
    "Excuse me, can you help me?",
    "The weather is very nice today.",
    "How much does this cost?",
    "I do not speak English.",
    "Where is the restroom?",
    "Could you please speak more slowly?",
    "Thank you very much for your help, that was really kind of you.",
]


def audio_duration_ms(wav_path):
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return round(info.duration * 1000, 2)
    except Exception as e:
        print("[WARN] Could not read duration: " + str(e))
        return 0


def run_pyttsx3(text, voice_identifier, output_path, rate=170):
    import pyttsx3
    t0 = time.perf_counter()
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    # best-effort voice match
    target = voice_identifier.lower()
    for v in engine.getProperty("voices"):
        hay = (str(v.id) + " " + str(v.name)).lower()
        if target in hay:
            engine.setProperty("voice", v.id)
            break
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    del engine
    synth_ms = round((time.perf_counter() - t0) * 1000, 2)
    return synth_ms


def run_piper(text, model_path, output_path, binary="piper"):
    safe = text.replace('"', '\\"')
    cmd = 'echo "' + safe + '" | ' + binary + ' -m ' + model_path + ' -f ' + output_path
    t0 = time.perf_counter()
    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    synth_ms = round((time.perf_counter() - t0) * 1000, 2)
    return synth_ms


def benchmark_engine(engine_name, sentences, output_dir, **kwargs):
    rows = []
    for i, sent in enumerate(sentences, start=1):
        out_path = os.path.join(output_dir, engine_name + "_" + str(i) + ".wav")
        try:
            if engine_name == "pyttsx3":
                synth_ms = run_pyttsx3(sent, kwargs["voice"], out_path,
                                       rate=kwargs.get("rate", 170))
            elif engine_name == "piper":
                synth_ms = run_piper(sent, kwargs["model"], out_path,
                                     binary=kwargs.get("binary", "piper"))
            else:
                raise ValueError("Unknown engine: " + engine_name)

            dur_ms = audio_duration_ms(out_path)
            size_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            rtf = round(synth_ms / dur_ms, 3) if dur_ms > 0 else ""

            rows.append({
                "engine": engine_name,
                "index": i,
                "sentence": sent,
                "word_count": len(sent.split()),
                "synth_ms": synth_ms,
                "audio_ms": dur_ms,
                "rtf": rtf,
                "file_bytes": size_bytes,
                "status": "ok",
                "error": "",
            })
            print("[" + engine_name + "] " + str(i) + "/" + str(len(sentences))
                  + ": synth=" + str(synth_ms) + "ms  audio=" + str(dur_ms) + "ms  RTF=" + str(rtf))

        except Exception as e:
            rows.append({
                "engine": engine_name, "index": i, "sentence": sent,
                "word_count": len(sent.split()),
                "synth_ms": "", "audio_ms": "", "rtf": "",
                "file_bytes": 0, "status": "error", "error": str(e),
            })
            print("[" + engine_name + "] " + str(i) + " ERROR: " + str(e))
    return rows


def summarize(rows):
    import statistics
    by_engine = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        by_engine.setdefault(r["engine"], []).append(r)

    lines = ["", "=" * 60, "TTS COMPARISON SUMMARY", "=" * 60]
    for eng, rs in by_engine.items():
        synths = [r["synth_ms"] for r in rs]
        rtfs = [r["rtf"] for r in rs if isinstance(r["rtf"], (int, float))]
        sizes = [r["file_bytes"] for r in rs]
        lines.append("")
        lines.append("Engine: " + eng + "  (n=" + str(len(rs)) + ")")
        lines.append("  Synthesis time (ms):  mean=" + ("%.1f" % statistics.mean(synths))
                     + "  median=" + ("%.1f" % statistics.median(synths))
                     + "  min=" + ("%.1f" % min(synths))
                     + "  max=" + ("%.1f" % max(synths)))
        if rtfs:
            lines.append("  RTF (synth/audio):    mean=" + ("%.3f" % statistics.mean(rtfs))
                         + "  median=" + ("%.3f" % statistics.median(rtfs)))
        lines.append("  Output file size (B): mean=" + ("%.0f" % statistics.mean(sizes)))
    lines.append("=" * 60)
    return "\n".join(lines)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sentences = SENTENCES_DE if args.lang == "de" else SENTENCES_EN

    all_rows = []

    if args.engine in ("pyttsx3", "both"):
        voice = "german" if args.lang == "de" else "english"
        all_rows += benchmark_engine(
            "pyttsx3", sentences, args.output_dir, voice=voice, rate=args.rate
        )

    if args.engine in ("piper", "both"):
        if not args.piper_model:
            print("[FATAL] --piper-model is required for Piper benchmark.")
            raise SystemExit(1)
        all_rows += benchmark_engine(
            "piper", sentences, args.output_dir,
            model=args.piper_model, binary=args.piper_binary
        )

    # CSV
    fields = ["engine", "index", "sentence", "word_count",
              "synth_ms", "audio_ms", "rtf", "file_bytes", "status", "error"]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\n[OK] CSV: " + args.csv)

    # Summary
    print(summarize(all_rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["pyttsx3", "piper", "both"], default="both")
    p.add_argument("--lang", choices=["de", "en"], default="en")
    p.add_argument("--rate", type=int, default=170, help="pyttsx3 rate (wpm)")
    p.add_argument("--piper-model", default=None, help="Path to Piper .onnx model")
    p.add_argument("--piper-binary", default="piper")
    p.add_argument("--output-dir", default="tts_out")
    p.add_argument("--csv", default="tts_compare.csv")
    args = p.parse_args()
    main(args)
