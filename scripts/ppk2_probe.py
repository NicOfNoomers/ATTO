#!/usr/bin/env python3
"""Bench probe for the Nordic PPK2 as an ammeter between the bench supply and the pump driver.

For bench testing only: in normal operation the dashboard runs the PPK2 itself
("Power meter (PPK2)" panel). Close the dashboard before running this.

Answers the questions the dashboard integration depends on:
  * Is the numpy decoder right? (compared against ppk2-api's own decoder)
  * Does the stream keep up at 100 kS/s, and can lost samples be detected
    (the 6-bit counter in each sample word)?
  * How much current/power does the pump draw per frequency, including 0 Hz?
  * (--clamp-test) How fast does the bench supply's current limit clamp?

Wiring (ammeter mode, the PPK2 never powers anything here):
  bench supply +  -> PPK2 VIN
  PPK2 VOUT       -> pump driver VM
  all grounds connected (bench -, PPK2 GND, driver GND)
  bench supply current limit set (60 mA for the pump)

Usage (C:\\Python313\\python.exe):
  python scripts/ppk2_probe.py --idle-check                       # first time: bench supply OFF, check VOUT ~ 0 V
  python scripts/ppk2_probe.py --supply-mv 2500 --arduino COM17   # 0/100/600/1140 Hz, 10 s each
  python scripts/ppk2_probe.py --supply-mv 2500 --seconds 20      # measure only, no pump commands
  python scripts/ppk2_probe.py --supply-mv 2500 --clamp-test --limit-ma 60 --seconds 20   # dummy load only!

Output: ppk2_probe_<time>/ with raw.bin (all sample words), meta.json,
summary.csv (one row per step) and probe.png.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# The decoder is shared with the dashboard's power meter (power_ppk2.py, repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from power_ppk2 import (FS as SAMPLE_RATE, best_alignment, connect_ppk2, counter_gaps,  # noqa: E402
                        decode_words, modifier_arrays)


# ---------------- hardware ----------------
def open_ppk2(port, vdd_mv):
    try:
        ppk, port = connect_ppk2(port)     # stops a leftover stream, reads the calibration
    except RuntimeError as e:
        sys.exit(f"{e}\nRun scripts\\check_setup.py to list the serial ports.")
    ppk.use_ampere_meter()
    # In ampere-meter mode this does not power anything: ppk2-api needs the
    # input voltage for its calibration (and refuses to measure without it).
    ppk.set_source_voltage(int(vdd_mv))
    return ppk, port


def open_arduino(port, baud):
    import serial
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.2)
    time.sleep(1.0)            # most boards reset on open
    if ser.in_waiting:
        ser.read(ser.in_waiting)
    return ser


def send_freq(ser, hz, protocol):
    if ser is None:
        return
    ser.write((f"{int(hz)}\n" if protocol == "freq_only" else f"F {int(hz)}\n").encode())
    ser.flush()


def acquire(ppk, steps, arduino, protocol):
    """Run the steps [(label, hz or None, seconds)], reading the PPK2 continuously.
    Returns (bytes, [(label, hz, byte_start, byte_end, t_start, t_end)], read stats)."""
    chunks, nbytes, marks = [], 0, []
    max_gap = 0.0
    ppk.start_measuring()
    t_last = time.monotonic()
    try:
        for label, hz, secs in steps:
            if hz is not None:
                send_freq(arduino, hz, protocol)
            b0, t0 = nbytes, time.monotonic()
            print(f"  {label:<14} {secs:5.1f} s ...", flush=True)
            while time.monotonic() - t0 < secs:
                d = ppk.get_data()
                now = time.monotonic()
                if d:
                    chunks.append(d)
                    nbytes += len(d)
                    max_gap = max(max_gap, now - t_last)
                    t_last = now
                time.sleep(0.002)
            marks.append((label, hz, b0, nbytes, t0, time.monotonic()))
    finally:
        send_freq(arduino, 0, protocol)
        ppk.stop_measuring()
    return b"".join(chunks), marks, {"max_read_gap_s": max_gap}


# ---------------- analysis ----------------
def analyse(data, marks, ppk, vdd_mv, limit_ma, clamp):
    marr = modifier_arrays(ppk.modifiers)
    off = best_alignment(data[:400_000])
    n = (len(data) - off) // 4
    words = np.frombuffer(data[off:off + 4 * n], dtype="<u4")
    t_dec = time.perf_counter()
    cur_ua, rng, counter, logic = decode_words(words, marr, vdd_mv)
    t_dec = time.perf_counter() - t_dec

    # Library decoder (pure Python, with its spike filter) on the first 0.5 s, for comparison
    k = min(n, SAMPLE_RATE // 2)
    ppk.remainder = {"sequence": b"", "len": 0}
    ppk.rolling_avg = ppk.rolling_avg4 = ppk.prev_range = None
    ppk.after_spike = ppk.consecutive_range_samples = 0
    t_lib = time.perf_counter()
    lib_ua, _bits = ppk.get_samples(data[off:off + 4 * k])
    t_lib = time.perf_counter() - t_lib
    lib_ua = np.asarray(lib_ua, dtype=float)
    m = min(lib_ua.size, k)
    diff = np.abs(cur_ua[:m] - lib_ua[:m])
    same = diff < 1e-6 * np.maximum(1.0, np.abs(lib_ua[:m]))

    rows = []
    for label, hz, b0, b1, t0, t1 in marks:
        i0, i1 = max(0, (b0 - off) // 4), max(0, (b1 - off) // 4)
        seg = cur_ua[i0:i1] / 1000.0          # mA
        if seg.size == 0:
            rows.append({"step": label, "freq_hz": hz, "samples": 0})
            continue
        dur = t1 - t0
        row = {
            "step": label, "freq_hz": "" if hz is None else hz, "seconds": round(dur, 3),
            "samples": int(seg.size), "rate_ksps": round(seg.size / dur / 1000, 2),
            "counter_gaps": counter_gaps(counter[i0:i1]),
            "i_mean_ma": round(float(seg.mean()), 4), "i_min_ma": round(float(seg.min()), 4),
            "i_max_ma": round(float(seg.max()), 4), "i_p99_ma": round(float(np.percentile(seg, 99)), 4),
            "p_mean_mw": round(vdd_mv / 1000.0 * float(seg.mean()), 3),
        }
        if clamp:
            over = seg > 1.2 * limit_ma
            edges = np.flatnonzero(np.diff(over.astype(np.int8)) == 1) + 1
            row["excursions_over_1.2x_limit"] = int(edges.size + (1 if over[:1].any() else 0))
            row["longest_over_ms"] = round(_longest_run(over) / SAMPLE_RATE * 1000, 3)
        rows.append(row)

    checks = {
        "alignment_offset_bytes": off,
        "decode_numpy_s_per_s_of_data": round(t_dec / max(n / SAMPLE_RATE, 1e-9), 5),
        "decode_library_s_per_s_of_data": round(t_lib / max(k / SAMPLE_RATE, 1e-9), 3),
        "library_vs_numpy_identical_fraction": round(float(np.mean(same)), 5) if m else None,
        "library_vs_numpy_mean_diff_ua": round(float(np.mean(lib_ua[:m] - cur_ua[:m])), 4) if m else None,
        "counter_in_use": bool(np.any(counter)),
        "ranges_seen": sorted(int(r) for r in np.unique(rng)),
        "logic_bits_seen": sorted(int(x) for x in np.unique(logic))[:16],
    }
    return cur_ua, rows, checks, off


def _longest_run(mask):
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(padded)
    return int(np.max(np.flatnonzero(d == -1) - np.flatnonzero(d == 1)))


def plot(cur_ua, marks, off, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(marks), 2, figsize=(13, 2.4 * len(marks)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 2]})
    for ax_row, (label, hz, b0, b1, t0, t1) in zip(axes, marks):
        i0, i1 = max(0, (b0 - off) // 4), max(0, (b1 - off) // 4)
        seg = cur_ua[i0:i1] / 1000.0
        if seg.size < 200:
            continue
        nb = seg.size // 100                                  # 1 ms bins
        b = seg[:nb * 100].reshape(nb, 100)
        t = np.arange(nb) / 1000.0
        ax_row[0].fill_between(t, b.min(1), b.max(1), color="0.8", lw=0)
        ax_row[0].plot(t, b.mean(1), lw=0.8)
        ax_row[0].set_ylabel("mA")
        ax_row[0].set_title(f"{label}: 1 ms mean (min-max shaded)", fontsize=9)
        z = seg[seg.size // 2: seg.size // 2 + 2000]          # 20 ms of raw samples mid-step
        ax_row[1].plot(np.arange(z.size) / 100.0, z, lw=0.7)
        ax_row[1].set_title("20 ms raw (100 kS/s)", fontsize=9)
    axes[-1][0].set_xlabel("s into step")
    axes[-1][1].set_xlabel("ms")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="PPK2 COM port (default: auto-detect)")
    ap.add_argument("--supply-mv", type=float, default=2500, help="bench supply voltage, for power and calibration")
    ap.add_argument("--arduino", help="pump Arduino COM port; without it nothing is sent to the pump")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--protocol", default="freq_only", choices=["freq_only", "freq_duty"])
    ap.add_argument("--freqs", default="0,100,600,1140", help="frequencies for the steps (Hz)")
    ap.add_argument("--step-s", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=None, help="measure-only duration (no --arduino)")
    ap.add_argument("--clamp-test", action="store_true", help="dummy load: report excursions above the limit")
    ap.add_argument("--limit-ma", type=float, default=60.0)
    ap.add_argument("--idle-check", action="store_true",
                    help="bench supply OFF: enable the PPK2 path so you can check VOUT reads ~0 V")
    ap.add_argument("--out", default=".", help="parent folder for the output")
    ap.add_argument("--yes", action="store_true", help="skip the wiring confirmation")
    a = ap.parse_args()

    if a.supply_mv > 5000 or a.supply_mv < 800:
        sys.exit("--supply-mv must be 800-5000 mV (PPK2 range).")

    if a.idle_check:
        ppk, port = open_ppk2(a.port, a.supply_mv)
        ppk.toggle_DUT_power("ON")
        print(f"PPK2 on {port}: ammeter mode, output path enabled. With the bench supply OFF, measure\n"
              "VOUT to GND with a multimeter now. It must read ~0 V (the PPK2 must not drive the line).")
        input("Press Enter when done ...")
        ppk.toggle_DUT_power("OFF")
        return

    if not a.yes:
        print("Check before starting:\n"
              f"  - bench supply at {a.supply_mv / 1000:.3f} V, current limit {a.limit_ma:g} mA\n"
              "  - bench + -> PPK2 VIN, PPK2 VOUT -> driver VM, all grounds connected\n"
              + ("  - DUMMY LOAD connected (not a pump)\n" if a.clamp_test else "")
              + ("  - nothing will be sent to the pump (no --arduino)\n" if not a.arduino else
                 f"  - the pump will be driven via {a.arduino}\n"))
        if input("Start? [y/N] ").strip().lower() != "y":
            return

    ppk, port = open_ppk2(a.port, a.supply_mv)
    arduino = open_arduino(a.arduino, a.baud) if a.arduino else None
    if a.arduino:
        steps = [(f"{hz} Hz", hz, a.step_s) for hz in (int(x) for x in a.freqs.split(","))]
    else:
        steps = [("measure", None, a.seconds or a.step_s)]
    out_dir = os.path.join(a.out, "ppk2_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)

    print(f"PPK2 {port}, ammeter mode, {a.supply_mv / 1000:.3f} V supply -> {out_dir}")
    ppk.toggle_DUT_power("ON")
    try:
        data, marks, stats = acquire(ppk, steps, arduino, a.protocol)
    finally:
        try:
            if arduino:
                arduino.close()
        except Exception:
            pass

    with open(os.path.join(out_dir, "raw.bin"), "wb") as f:
        f.write(data)
    cur_ua, rows, checks, off = analyse(data, marks, ppk, a.supply_mv, a.limit_ma, a.clamp_test)
    checks.update(stats)
    meta = {"port": port, "supply_mv": a.supply_mv, "limit_ma": a.limit_ma, "args": vars(a),
            "modifiers": ppk.modifiers, "steps": [dict(zip(("label", "hz", "byte_start", "byte_end",
                                                            "t_start", "t_end"), m)) for m in marks],
            "checks": checks}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    plot(cur_ua, marks, off, os.path.join(out_dir, "probe.png"),
         f"PPK2 probe {os.path.basename(out_dir)}, supply {a.supply_mv / 1000:.3f} V")

    print("\nper step:")
    for r in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
