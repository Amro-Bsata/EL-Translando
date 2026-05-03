"""
client_benchmark.py - text translation benchmark (runs on client / laptop).

Sends text-only translation requests to the server and measures:
    - end-to-end latency (network + server)
    - server-only latency (from server's timing report)
    - network overhead = end_to_end - server_total
    - latency grouped by sentence-length bucket

Usage:
    python3 client_benchmark.py --url http://192.168.1.42:5000
    python3 client_benchmark.py --direction en2de --repeats 5
    python3 client_benchmark.py --corpus client/corpus_de.json
"""

import argparse
import os
import sys
import time
import requests

from client_common import (
    load_corpus, wait_for_server, summarize, write_csv,
    aggregate_by_bucket, BUCKET_ORDER,
)


def send_text(url, text, direction, timeout, tts_engine="pyttsx3"):
    payload = {"text": text, "direction": direction, "tts_engine": tts_engine}
    t0 = time.perf_counter()
    err = None
    data = None
    try:
        r = requests.post(url + "/translate/text", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        err = e
    end_to_end_ms = round((time.perf_counter() - t0) * 1000, 2)
    return data, err, end_to_end_ms


def run(args):
    print("[INFO] Server: " + args.url)
    print("[INFO] Corpus: " + args.corpus)
    print("[INFO] Direction: " + args.direction)
    print("[INFO] TTS engine: " + args.tts_engine)
    print("[INFO] Repeats: " + str(args.repeats))
    print("")

    if not wait_for_server(args.url, timeout=args.wait):
        print("[FATAL] Server not reachable.")
        sys.exit(1)

    lang, sentences = load_corpus(args.corpus)
    print("[INFO] Loaded " + str(len(sentences)) + " sentences (" + lang + ")")
    total_requests = len(sentences) * args.repeats
    print("[INFO] Will send " + str(total_requests) + " requests total")
    print("")

    results = []
    for rep in range(args.repeats):
        print("--- Repeat " + str(rep + 1) + "/" + str(args.repeats) + " ---")
        for i, item in enumerate(sentences, start=1):
            data, err, e2e_ms = send_text(
                args.url, item["sentence"], args.direction, args.timeout,
                tts_engine=args.tts_engine,
            )

            row = {
                "repeat": rep + 1,
                "index": i,
                "bucket": item["bucket"],
                "word_count": item["word_count"],
                "sentence": item["sentence"],
                "tts_engine": args.tts_engine,
                "status": "error" if err else "ok",
                "end_to_end_ms": e2e_ms,
            }

            if data and data.get("status") == "success":
                t = data.get("timings", {})
                row["mt_ms"] = t.get("mt_ms")
                row["tts_ms"] = t.get("tts_ms")
                row["server_total_ms"] = t.get("total_ms")
                row["audio_duration_ms"] = t.get("audio_duration_ms")
                row["network_overhead_ms"] = round(
                    e2e_ms - (t.get("total_ms") or 0), 2
                )
                row["translated"] = data.get("translated", "")
                row["error"] = ""
            else:
                row.update({
                    "mt_ms": "", "tts_ms": "", "server_total_ms": "",
                    "audio_duration_ms": "", "network_overhead_ms": "",
                    "translated": "", "error": str(err) if err else "",
                })

            results.append(row)

            mark = "." if row["status"] == "ok" else "X"
            sys.stdout.write(mark)
            sys.stdout.flush()
        print("")

    # ============ Write per-request CSV ============
    fields = ["repeat", "index", "bucket", "word_count", "sentence",
              "tts_engine",
              "status", "end_to_end_ms", "mt_ms", "tts_ms",
              "server_total_ms", "network_overhead_ms",
              "audio_duration_ms", "translated", "error"]
    write_csv(args.output, results, fields)
    print("\n[OK] Per-request CSV: " + args.output)

    # ============ Bucket aggregation ============
    bucket_rows_e2e = aggregate_by_bucket(results, "end_to_end_ms")
    bucket_rows_srv = aggregate_by_bucket(results, "server_total_ms")
    bucket_rows_net = aggregate_by_bucket(results, "network_overhead_ms")

    bucket_csv = args.output.replace(".csv", "_by_length.csv")
    with_field = []
    for i, b in enumerate(bucket_rows_e2e):
        with_field.append({
            "bucket": b["bucket"],
            "n_requests": b["n"],
            "end_to_end_mean": b["mean"],
            "end_to_end_median": b["median"],
            "end_to_end_p95": b["p95"],
            "server_total_mean": bucket_rows_srv[i]["mean"],
            "server_total_median": bucket_rows_srv[i]["median"],
            "network_overhead_mean": bucket_rows_net[i]["mean"],
        })
    write_csv(bucket_csv, with_field, [
        "bucket", "n_requests",
        "end_to_end_mean", "end_to_end_median", "end_to_end_p95",
        "server_total_mean", "server_total_median",
        "network_overhead_mean",
    ])
    print("[OK] Length-bucket CSV: " + bucket_csv)

    # ============ Summary ============
    summary = build_summary(results, bucket_rows_e2e, bucket_rows_srv,
                            bucket_rows_net, args)
    print("\n" + summary)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary)
    print("[OK] Summary: " + args.summary)


def build_summary(rows, e2e, srv, net, args):
    ok = [r for r in rows if r["status"] == "ok"]
    total = len(rows)
    errors = total - len(ok)

    e2e_vals = [r["end_to_end_ms"] for r in ok]
    srv_vals = [r["server_total_ms"] for r in ok if isinstance(r["server_total_ms"], (int, float))]
    mt_vals  = [r["mt_ms"] for r in ok if isinstance(r["mt_ms"], (int, float))]
    tts_vals = [r["tts_ms"] for r in ok if isinstance(r["tts_ms"], (int, float))]
    net_vals = [r["network_overhead_ms"] for r in ok if isinstance(r["network_overhead_ms"], (int, float))]

    lines = []
    lines.append("=" * 70)
    lines.append("CLIENT BENCHMARK SUMMARY (text translation)")
    lines.append("=" * 70)
    lines.append("Server:        " + args.url)
    lines.append("Direction:     " + args.direction)
    lines.append("Total req:     " + str(total))
    lines.append("Successful:    " + str(len(ok)))
    lines.append("Errors:        " + str(errors)
                 + "  (" + ("%.2f" % (100.0 * errors / total)) + "%)")
    lines.append("")
    lines.append(summarize(e2e_vals, "End-to-end latency  "))
    lines.append(summarize(srv_vals, "Server-only latency "))
    lines.append(summarize(net_vals, "Network overhead    "))
    lines.append(summarize(mt_vals,  "  - MT stage        "))
    lines.append(summarize(tts_vals, "  - TTS stage       "))
    lines.append("")
    lines.append("LATENCY BY SENTENCE LENGTH (end-to-end, ms):")
    lines.append("  " + "bucket".ljust(13) + "n".rjust(5)
                 + "mean".rjust(10) + "median".rjust(10)
                 + "p95".rjust(10) + "min".rjust(10) + "max".rjust(10))
    for b in e2e:
        lines.append("  " + b["bucket"].ljust(13)
                     + str(b["n"]).rjust(5)
                     + str(b["mean"]).rjust(10)
                     + str(b["median"]).rjust(10)
                     + str(b["p95"]).rjust(10)
                     + str(b["min"]).rjust(10)
                     + str(b["max"]).rjust(10))
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5000")
    p.add_argument("--corpus", default="corpus_de.json")
    p.add_argument("--direction", choices=["de2en", "en2de"], default="de2en")
    p.add_argument("--tts-engine", dest="tts_engine",
                   choices=["pyttsx3", "piper"], default="pyttsx3",
                   help="Which TTS engine the server should use")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--wait", type=int, default=180)
    p.add_argument("--output", default="text_results.csv")
    p.add_argument("--summary", default="text_summary.txt")
    args = p.parse_args()
    run(args)
