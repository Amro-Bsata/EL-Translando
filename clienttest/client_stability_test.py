"""
client_stability_test.py - long-running stability test (runs on client).

Sends N requests against the server and analyzes:
    - errors per 100 requests
    - latency drift (mean / p95 per bucket of 100)
    - server downtime detection
    - error categorization

Usage:
    python3 client_stability_test.py --url http://192.168.1.42:5000 \\
                                     --requests 500 --mixed
"""

import argparse
import statistics
import sys
import time
import requests

from client_common import wait_for_server, write_csv, percentile, load_corpus


def categorize_error(exc, data):
    if exc is None and data and data.get("status") == "success":
        return "ok"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection_refused"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.HTTPError):
        return "http_error"
    if isinstance(exc, ValueError):
        return "malformed_json"
    if data and data.get("detail"):
        msg = str(data["detail"]).lower()
        if "stt" in msg:
            return "stt_failed"
        if "tts" in msg:
            return "tts_failed"
        return "server_error"
    return "unknown"


def do_request(url, text, direction, timeout, tts_engine="pyttsx3"):
    t0 = time.perf_counter()
    err = None
    data = None
    try:
        r = requests.post(url + "/translate/text",
                          json={"text": text, "direction": direction,
                                "tts_engine": tts_engine},
                          timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = None
        if r.status_code >= 400:
            err = requests.exceptions.HTTPError("HTTP " + str(r.status_code))
    except Exception as e:
        err = e
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"data": data, "error": err, "wall_ms": wall_ms,
            "category": categorize_error(err, data)}


def server_alive(url):
    try:
        r = requests.get(url + "/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run(args):
    print("[INFO] Stability test against " + args.url)
    print("[INFO] Total requests: " + str(args.requests))
    print("[INFO] Bucket size: " + str(args.bucket))
    print("[INFO] TTS engine: " + args.tts_engine)
    print("")

    if not wait_for_server(args.url, timeout=args.wait):
        print("[FATAL] Server not reachable.")
        sys.exit(1)

    # Load both corpora
    _, sentences_de = load_corpus(args.corpus_de)
    _, sentences_en = load_corpus(args.corpus_en)

    rows = []
    downtimes = []
    in_down = False
    down_start = None
    consec_conn = 0
    test_start = time.time()

    for i in range(args.requests):
        if args.mixed:
            direction = "de2en" if (i % 2 == 0) else "en2de"
            sentences = sentences_de if direction == "de2en" else sentences_en
        else:
            direction = args.direction
            sentences = sentences_de if direction == "de2en" else sentences_en

        item = sentences[i % len(sentences)]
        res = do_request(args.url, item["sentence"], direction, args.timeout,
                         tts_engine=args.tts_engine)

        if res["category"] == "connection_refused":
            consec_conn += 1
            if consec_conn == 3 and not in_down:
                in_down = True
                down_start = time.time()
                print("\n[ALERT] Server appears DOWN at request #" + str(i + 1))
        else:
            if in_down:
                downtimes.append((down_start, time.time()))
                dur = round(time.time() - down_start, 1)
                print("\n[INFO] Server recovered at #" + str(i + 1)
                      + " (downtime " + str(dur) + "s)")
                in_down = False
            consec_conn = 0

        timings = (res["data"] or {}).get("timings", {}) if res["data"] else {}
        rows.append({
            "request_index": i + 1,
            "elapsed_s": round(time.time() - test_start, 1),
            "direction": direction,
            "bucket": item["bucket"],
            "word_count": item["word_count"],
            "category": res["category"],
            "client_wall_ms": res["wall_ms"],
            "mt_ms": timings.get("mt_ms", ""),
            "tts_ms": timings.get("tts_ms", ""),
            "server_total_ms": timings.get("total_ms", ""),
            "error_message": str(res["error"])[:200] if res["error"] else "",
        })

        if (i + 1) % 25 == 0:
            recent = rows[-25:]
            ok_count = sum(1 for r in recent if r["category"] == "ok")
            avg_lat = statistics.mean([r["client_wall_ms"] for r in recent])
            print("[" + str(i + 1).rjust(4) + "/" + str(args.requests) + "]  "
                  + str(ok_count) + "/25 ok  "
                  + "avg=" + ("%.0f" % avg_lat) + "ms")

        if args.delay > 0:
            time.sleep(args.delay)

    if in_down:
        downtimes.append((down_start, time.time()))

    # Per-request CSV
    write_csv(args.output, rows, [
        "request_index", "elapsed_s", "direction", "bucket", "word_count",
        "category", "client_wall_ms", "mt_ms", "tts_ms", "server_total_ms",
        "error_message",
    ])

    # Buckets
    buckets = build_buckets(rows, args.bucket)
    write_csv(args.buckets, buckets, [
        "bucket_index", "request_range", "total", "ok", "errors",
        "errors_per_100", "mean_ms", "median_ms", "p95_ms", "max_ms",
    ])

    # Summary
    summary = build_summary(rows, buckets, downtimes, args)
    print("\n" + summary)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n[OK] Files written: " + args.output + ", " + args.buckets
          + ", " + args.summary)


def build_buckets(rows, size):
    out = []
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        ok = [r for r in chunk if r["category"] == "ok"]
        errs = len(chunk) - len(ok)
        lat = [r["client_wall_ms"] for r in ok]
        if lat:
            mean_v, med_v, p95_v, max_v = (
                statistics.mean(lat), statistics.median(lat),
                percentile(lat, 95), max(lat),
            )
        else:
            mean_v = med_v = p95_v = max_v = 0
        out.append({
            "bucket_index": len(out) + 1,
            "request_range": str(start + 1) + "-" + str(start + len(chunk)),
            "total": len(chunk),
            "ok": len(ok),
            "errors": errs,
            "errors_per_100": round(100.0 * errs / len(chunk), 2),
            "mean_ms": round(mean_v, 1),
            "median_ms": round(med_v, 1),
            "p95_ms": round(p95_v, 1),
            "max_ms": round(max_v, 1),
        })
    return out


def build_summary(rows, buckets, downtimes, args):
    total = len(rows)
    ok = sum(1 for r in rows if r["category"] == "ok")
    errs = total - ok
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    drift_msg = ""
    if len(buckets) >= 2 and buckets[0]["mean_ms"] > 0:
        first = buckets[0]["mean_ms"]
        last = buckets[-1]["mean_ms"]
        pct = round(100.0 * (last - first) / first, 1)
        drift_msg = ("Latency drift first->last bucket: "
                     + str(first) + "ms -> " + str(last) + "ms  ("
                     + ("+" if pct >= 0 else "") + str(pct) + "%)")

    L = []
    L.append("=" * 70)
    L.append("STABILITY TEST SUMMARY")
    L.append("=" * 70)
    L.append("Total:         " + str(total))
    L.append("Successful:    " + str(ok))
    L.append("Errors:        " + str(errs))
    L.append("Errors / 100:  " + ("%.2f" % (100.0 * errs / total)))
    L.append("")
    L.append("Error categories:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        L.append("  " + cat.ljust(22) + str(n).rjust(6))
    L.append("")
    if downtimes:
        L.append("Server downtimes: " + str(len(downtimes)))
        for i, (s, e) in enumerate(downtimes, 1):
            L.append("  #" + str(i) + ": " + ("%.1f" % (e - s)) + "s")
    else:
        L.append("Server downtimes: 0")
    L.append("")
    if drift_msg:
        L.append(drift_msg)
    L.append("")
    L.append("Per-bucket overview:")
    L.append("  " + "bucket".ljust(8) + "errors/100".ljust(13)
             + "mean(ms)".ljust(12) + "p95(ms)".ljust(12) + "max(ms)")
    for b in buckets:
        L.append("  " + str(b["bucket_index"]).ljust(8)
                 + str(b["errors_per_100"]).ljust(13)
                 + str(b["mean_ms"]).ljust(12)
                 + str(b["p95_ms"]).ljust(12)
                 + str(b["max_ms"]))
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5000")
    p.add_argument("--corpus-de", default="corpus_de.json")
    p.add_argument("--corpus-en", default="corpus_en.json")
    p.add_argument("--direction", choices=["de2en", "en2de"], default="de2en")
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--tts-engine", dest="tts_engine",
                   choices=["pyttsx3", "piper"], default="pyttsx3")
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--bucket", type=int, default=100)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--wait", type=int, default=180)
    p.add_argument("--output", default="stability_results.csv")
    p.add_argument("--buckets", default="stability_buckets.csv")
    p.add_argument("--summary", default="stability_summary.txt")
    args = p.parse_args()
    run(args)
