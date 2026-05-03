"""
resource_monitor.py

Logs CPU, RAM, temperature and CPU frequency once per second to a CSV file.
Run this in a separate terminal on the UNO Q while the benchmark is running.

Usage:
    python3 resource_monitor.py --duration 600 --output resources.csv
    python3 resource_monitor.py --duration 0          # runs forever until Ctrl-C

Install:
    pip install psutil
"""

import argparse
import csv
import glob
import os
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    print("[FATAL] psutil not installed. Run: pip install psutil")
    raise SystemExit(1)


NOMINAL_FREQ_KHZ = 2_000_000   # 2.0 GHz for Cortex-A53 on QRB2210
THROTTLE_RATIO = 0.90          # below 90% of nominal = throttled


def discover_thermal_zones():
    zones = []
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(z, "type")) as f:
                ztype = f.read().strip()
            zones.append((ztype, os.path.join(z, "temp")))
        except Exception:
            pass
    return zones


def discover_cpu_freqs():
    cpus = []
    for c in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*")):
        freq_path = os.path.join(c, "cpufreq/scaling_cur_freq")
        if os.path.exists(freq_path):
            cpus.append((os.path.basename(c), freq_path))
    return cpus


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def main(args):
    zones = discover_thermal_zones()
    cpus = discover_cpu_freqs()

    if zones:
        print("[INFO] Thermal zones: " + ", ".join(z[0] for z in zones))
    else:
        print("[WARN] No thermal zones found - temperature will be empty.")
    print("[INFO] CPUs tracked: " + ", ".join(c[0] for c in cpus))
    print("[INFO] Sampling every " + str(args.interval) + "s "
          + ("forever" if args.duration == 0 else "for " + str(args.duration) + "s"))

    headers = ["timestamp", "elapsed_s",
               "cpu_total_percent", "cpu_per_core_percent",
               "ram_used_mb", "ram_total_mb", "ram_percent",
               "swap_used_mb"]
    headers += ["temp_" + z[0] + "_C" for z in zones]
    headers += ["freq_" + c[0] + "_MHz" for c in cpus]
    headers += ["throttled"]

    throttle_events = 0
    start = time.time()

    # Prime cpu_percent() - first call returns 0.0
    psutil.cpu_percent(interval=None, percpu=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        try:
            while True:
                t_sample = time.time()
                elapsed = t_sample - start

                if args.duration > 0 and elapsed >= args.duration:
                    break

                # CPU (blocks for `interval` seconds and returns averaged value)
                per_core = psutil.cpu_percent(interval=args.interval, percpu=True)
                total = sum(per_core) / len(per_core) if per_core else 0.0

                # RAM
                vm = psutil.virtual_memory()
                sw = psutil.swap_memory()

                # Temperatures
                temps = []
                for _, tpath in zones:
                    v = read_int(tpath)
                    temps.append(v / 1000.0 if v is not None else None)

                # Frequencies
                freqs_khz = [read_int(c[1]) for c in cpus]
                freqs_mhz = [(fr / 1000.0 if fr is not None else None) for fr in freqs_khz]

                # Throttling detection
                valid = [fr for fr in freqs_khz if fr is not None]
                is_throttled = (
                    any(fr < NOMINAL_FREQ_KHZ * THROTTLE_RATIO for fr in valid)
                    if valid else False
                )
                if is_throttled:
                    throttle_events += 1

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    round(elapsed, 1),
                    round(total, 1),
                    "|".join(("%.1f" % c) for c in per_core),
                    round(vm.used / 1e6, 1),
                    round(vm.total / 1e6, 1),
                    round(vm.percent, 1),
                    round(sw.used / 1e6, 1),
                ] + [
                    ("%.1f" % t) if t is not None else "" for t in temps
                ] + [
                    ("%.0f" % fr) if fr is not None else "" for fr in freqs_mhz
                ] + [
                    int(is_throttled)
                ])
                f.flush()

                # Live status every 5 samples
                if int(elapsed) % 5 == 0 or is_throttled:
                    max_temp = max((t for t in temps if t is not None), default=0)
                    min_freq = min(valid, default=0) / 1000.0
                    print(("[%5.0fs] CPU=%5.1f%%  RAM=%5.1f%%  "
                           "maxT=%4.1fC  minFreq=%5.0fMHz  "
                           "throttle_events=%d") % (
                        elapsed, total, vm.percent,
                        max_temp, min_freq, throttle_events
                    ))

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")

    print("\n[DONE] CSV: " + args.output)
    print("[DONE] Throttle events: " + str(throttle_events))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=600, help="Duration in seconds (0 = forever)")
    p.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    p.add_argument("--output", default="resources.csv")
    args = p.parse_args()
    main(args)
