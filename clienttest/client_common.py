"""
client_common.py - shared helpers for all client test scripts.
"""

import csv
import json
import os
import statistics
import time
import requests


# ============================================================
# Corpus loading
# ============================================================
def load_corpus(path):
    """Load a corpus JSON and flatten into list of (bucket, sentence)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    flat = []
    for bucket_name, bucket in data["buckets"].items():
        for s in bucket["sentences"]:
            flat.append({
                "bucket": bucket_name,
                "word_range": bucket["word_range"],
                "sentence": s,
                "word_count": len(s.split()),
            })
    return data["language"], flat


# ============================================================
# Server health
# ============================================================
def wait_for_server(url, timeout=180, verbose=True):
    """Wait until /health returns 200 or timeout. Returns True/False."""
    for i in range(timeout):
        try:
            r = requests.get(url + "/health", timeout=3)
            if r.status_code == 200:
                if verbose:
                    print("[OK] Server reachable after " + str(i) + "s")
                return True
        except Exception:
            pass
        if verbose and i % 10 == 0 and i > 0:
            print("[WAIT] Still waiting for server... (" + str(i) + "s)")
        time.sleep(1)
    return False


# ============================================================
# Statistics
# ============================================================
def percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(values, label, unit="ms"):
    """Return formatted statistics line."""
    if not values:
        return label + ": no data"
    return (label + " (" + unit + "): "
            + "n=" + str(len(values))
            + "  mean=" + ("%.1f" % statistics.mean(values))
            + "  median=" + ("%.1f" % statistics.median(values))
            + "  p95=" + ("%.1f" % percentile(values, 95))
            + "  min=" + ("%.1f" % min(values))
            + "  max=" + ("%.1f" % max(values)))


# ============================================================
# CSV writing
# ============================================================
def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Length-bucket aggregation
# ============================================================
BUCKET_ORDER = ["very_short", "short", "medium", "long", "very_long"]


def aggregate_by_bucket(rows, value_field, bucket_field="bucket"):
    """Group rows by bucket and compute statistics for value_field."""
    by_bucket = {b: [] for b in BUCKET_ORDER}
    for r in rows:
        if r.get("status") != "ok":
            continue
        v = r.get(value_field)
        if isinstance(v, (int, float)):
            by_bucket[r[bucket_field]].append(v)

    result = []
    for bucket in BUCKET_ORDER:
        vals = by_bucket[bucket]
        if vals:
            result.append({
                "bucket": bucket,
                "n": len(vals),
                "mean": round(statistics.mean(vals), 1),
                "median": round(statistics.median(vals), 1),
                "p95": round(percentile(vals, 95), 1),
                "min": round(min(vals), 1),
                "max": round(max(vals), 1),
            })
        else:
            result.append({
                "bucket": bucket, "n": 0,
                "mean": "", "median": "", "p95": "", "min": "", "max": "",
            })
    return result
