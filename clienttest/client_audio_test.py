"""
client_audio_test.py - audio translation benchmark (runs on client / laptop).

Two modes:
    --mode wav     : uses pre-recorded WAV files from a folder
                     (filename pattern: <bucket>_<index>.wav, e.g. short_01.wav)
                     This is the reproducible mode, recommended for the thesis.

    --mode tts     : generates audio on the fly via pyttsx3 from the corpus.
                     Faster to set up but TTS audio is unrealistic (clean,
                     no noise, perfect articulation). Use only as a baseline.

Same metrics as client_benchmark.py PLUS:
    - input audio duration
    - speech-to-text latency (separate stage)

Usage:
    # Mode 1: pre-recorded WAVs (you record once, replay forever)
    python3 client_audio_test.py --mode wav --audio-dir ./audio_de \\
                                 --corpus corpus_de.json

    # Mode 2: TTS-generated (auto-generates audios from corpus)
    python3 client_audio_test.py --mode tts --corpus corpus_de.json
"""

import argparse
import os
import sys
import tempfile
import time
import requests

from client_common import (
    load_corpus, wait_for_server, summarize, write_csv,
    aggregate_by_bucket, BUCKET_ORDER,
)


# ============================================================
# Audio source helpers
# ============================================================
def find_wav_for(item, audio_dir, index):
    """Locate a WAV file for a given corpus item."""
    bucket = item["bucket"]
    candidates = [
        os.path.join(audio_dir, bucket + "_" + str(index).zfill(2) + ".wav"),
        os.path.join(audio_dir, bucket + "_" + str(index) + ".wav"),
        os.path.join(audio_dir, bucket, str(index) + ".wav"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def generate_tts_audio(text, lang, output_path):
    """Generate a WAV from text via pyttsx3 (client-side)."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    target = "german" if lang == "de" else "english"
    for v in engine.getProperty("voices"):
        hay = (str(v.id) + " " + str(v.name)).lower()
        if target in hay:
            engine.setProperty("voice", v.id)
            break
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    del engine
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def get_audio_duration_ms(path):
    try:
        import soundfile as sf
        return round(sf.info(path).duration * 1000, 2)
    except Exception:
        return 0


# ============================================================
# Single request
# ============================================================
def send_audio(url, audio_path, direction, timeout, tts_engine="pyttsx3"):
    t0 = time.perf_counter()
    err = None
    data = None
    try:
        with open(audio_path, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_path), f, "audio/wav")}
            payload = {"direction": direction, "tts_engine": tts_engine}
            r = requests.post(url + "/translate/audio",
                              files=files, data=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        err = e
    e2e_ms = round((time.perf_counter() - t0) * 1000, 2)
    return data, err, e2e_ms


# ============================================================
# Main
# ============================================================
def run(args):
    print("[INFO] Server:    " + args.url)
    print("[INFO] Mode:      " + args.mode)
    print("[INFO] Corpus:    " + args.corpus)
    print("[INFO] Direction: " + args.direction)
    print("[INFO] TTS engine: " + args.tts_engine)
    print("")

    if not wait_for_server(args.url, timeout=args.wait):
        print("[FATAL] Server not reachable.")
        sys.exit(1)

    lang, sentences = load_corpus(args.corpus)
    print("[INFO] " + str(len(sentences)) + " sentences loaded (" + lang + ")")

    if args.mode == "wav":
        if not args.audio_dir or not os.path.isdir(args.audio_dir):
            print("[FATAL] --audio-dir is required and must exist for wav mode")
            sys.exit(1)
        print("[INFO] Audio source: " + args.audio_dir)

    if args.mode == "tts":
        tmp_dir = tempfile.mkdtemp(prefix="client_tts_")
        print("[INFO] Generating TTS audios in: " + tmp_dir)

    print("")

    results = []
    for rep in range(args.repeats):
        print("--- Repeat " + str(rep + 1) + "/" + str(args.repeats) + " ---")
        for i, item in enumerate(sentences, start=1):
            # Resolve audio path
            if args.mode == "wav":
                audio_path = find_wav_for(item, args.audio_dir, i)
                if not audio_path:
                    print("\n[SKIP] No WAV found for bucket=" + item["bucket"]
                          + " index=" + str(i))
                    continue
            else:
                audio_path = os.path.join(tmp_dir, "tts_" + str(i) + ".wav")
                if not os.path.exists(audio_path):
                    if not generate_tts_audio(item["sentence"], lang, audio_path):
                        print("\n[SKIP] TTS generation failed for #" + str(i))
                        continue

            input_dur_ms = get_audio_duration_ms(audio_path)

            data, err, e2e_ms = send_audio(
                args.url, audio_path, args.direction, args.timeout,
                tts_engine=args.tts_engine,
            )

            row = {
                "repeat": rep + 1,
                "index": i,
                "bucket": item["bucket"],
                "word_count": item["word_count"],
                "sentence": item["sentence"],
                "audio_path": audio_path,
                "tts_engine": args.tts_engine,
                "input_audio_ms": input_dur_ms,
                "status": "error" if err else "ok",
                "end_to_end_ms": e2e_ms,
            }

            if data and data.get("status") == "success":
                t = data.get("timings", {})
                row["stt_ms"] = t.get("stt_ms")
                row["mt_ms"] = t.get("mt_ms")
                row["tts_ms"] = t.get("tts_ms")
                row["server_total_ms"] = t.get("total_ms")
                row["output_audio_ms"] = t.get("audio_duration_ms")
                row["network_overhead_ms"] = round(
                    e2e_ms - (t.get("total_ms") or 0), 2
                )
                row["transcribed"] = data.get("original", "")
                row["translated"] = data.get("translated", "")
                row["error"] = ""
            else:
                row.update({
                    "stt_ms": "", "mt_ms": "", "tts_ms": "",
                    "server_total_ms": "", "output_audio_ms": "",
                    "network_overhead_ms": "",
                    "transcribed": "", "translated": "",
                    "error": str(err) if err else "",
                })

            results.append(row)
            sys.stdout.write("." if row["status"] == "ok" else "X")
            sys.stdout.flush()
        print("")

    # CSV
    fields = ["repeat", "index", "bucket", "word_count", "sentence",
              "audio_path", "tts_engine",
              "status", "input_audio_ms", "end_to_end_ms",
              "stt_ms", "mt_ms", "tts_ms", "server_total_ms",
              "network_overhead_ms", "output_audio_ms",
              "transcribed", "translated", "error"]
    write_csv(args.output, results, fields)
    print("\n[OK] CSV: " + args.output)

    # Bucket aggregation by length
    e2e = aggregate_by_bucket(results, "end_to_end_ms")
    stt = aggregate_by_bucket(results, "stt_ms")

    bucket_csv = args.output.replace(".csv", "_by_length.csv")
    rows = []
    for i, b in enumerate(e2e):
        rows.append({
            "bucket": b["bucket"],
            "n": b["n"],
            "end_to_end_mean": b["mean"],
            "end_to_end_median": b["median"],
            "end_to_end_p95": b["p95"],
            "stt_mean": stt[i]["mean"],
            "stt_p95": stt[i]["p95"],
        })
    write_csv(bucket_csv, rows, [
        "bucket", "n", "end_to_end_mean", "end_to_end_median",
        "end_to_end_p95", "stt_mean", "stt_p95",
    ])
    print("[OK] Length-bucket CSV: " + bucket_csv)

    # Summary
    summary = build_summary(results, e2e, args)
    print("\n" + summary)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary)


def build_summary(rows, e2e, args):
    ok = [r for r in rows if r["status"] == "ok"]
    total = len(rows)
    errors = total - len(ok)

    e2e_vals = [r["end_to_end_ms"] for r in ok]
    stt_vals = [r["stt_ms"] for r in ok if isinstance(r["stt_ms"], (int, float))]
    mt_vals  = [r["mt_ms"]  for r in ok if isinstance(r["mt_ms"], (int, float))]
    tts_vals = [r["tts_ms"] for r in ok if isinstance(r["tts_ms"], (int, float))]

    lines = []
    lines.append("=" * 70)
    lines.append("CLIENT AUDIO BENCHMARK SUMMARY")
    lines.append("=" * 70)
    lines.append("Mode:          " + args.mode)
    lines.append("Server:        " + args.url)
    lines.append("Direction:     " + args.direction)
    lines.append("Total req:     " + str(total))
    lines.append("Successful:    " + str(len(ok)))
    lines.append("Errors:        " + str(errors)
                 + "  (" + ("%.2f" % (100.0 * errors / max(1, total))) + "%)")
    lines.append("")
    lines.append(summarize(e2e_vals, "End-to-end latency"))
    lines.append(summarize(stt_vals, "  - STT stage     "))
    lines.append(summarize(mt_vals,  "  - MT stage      "))
    lines.append(summarize(tts_vals, "  - TTS stage     "))
    lines.append("")
    lines.append("LATENCY BY SENTENCE LENGTH (end-to-end, ms):")
    lines.append("  " + "bucket".ljust(13) + "n".rjust(5)
                 + "mean".rjust(10) + "median".rjust(10)
                 + "p95".rjust(10))
    for b in e2e:
        lines.append("  " + b["bucket"].ljust(13)
                     + str(b["n"]).rjust(5)
                     + str(b["mean"]).rjust(10)
                     + str(b["median"]).rjust(10)
                     + str(b["p95"]).rjust(10))
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5000")
    p.add_argument("--mode", choices=["wav", "tts"], default="wav")
    p.add_argument("--corpus", default="corpus_de.json")
    p.add_argument("--direction", choices=["de2en", "en2de"], default="de2en")
    p.add_argument("--tts-engine", dest="tts_engine",
                   choices=["pyttsx3", "piper"], default="pyttsx3",
                   help="Which TTS engine the server should use")
    p.add_argument("--audio-dir", default=None,
                   help="Folder with pre-recorded WAVs (mode=wav)")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--wait", type=int, default=180)
    p.add_argument("--output", default="audio_results.csv")
    p.add_argument("--summary", default="audio_summary.txt")
    args = p.parse_args()
    run(args)
