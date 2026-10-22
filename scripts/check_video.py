#!/usr/bin/env python3
"""Check a run folder's video log: frames.csv against the video files.

    python scripts/check_video.py <run folder>

Prints, and flags with FAIL:
  * frames.csv rows vs frames decoded from video_part*.avi (must be equal, per file too)
  * frame rate from frames.csv vs the rate in the video header (within 5 %)
  * dropped frames (dropped_before; must be 0) and gaps > 1.5 frame intervals
  * the largest gap between frames (must be < 3 intervals)
  * frames per 1 Hz drive (from events.csv), if the run has them
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np


def main(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"))))
    files = sorted(glob.glob(os.path.join(run_dir, "video_part*")))
    fails = []
    print(f"{run_dir}\n  frames.csv rows: {len(rows)}, video files: {len(files)}")
    per_file = {}
    for r in rows:
        per_file[r["video_file"]] = per_file.get(r["video_file"], 0) + 1
    total, hdr = 0, []
    for f in files:
        cap = cv2.VideoCapture(f)
        n = 0
        while cap.read()[0]:
            n += 1
        hdr.append(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        total += n
        if n != per_file.get(os.path.basename(f), 0):
            fails.append(f"{os.path.basename(f)}: {n} decoded vs {per_file.get(os.path.basename(f), 0)} rows")
    print(f"  decoded video frames: {total}")
    if total != len(rows):
        fails.append(f"frames.csv rows {len(rows)} != decoded frames {total}")
    t = np.array([float(r["t_s"]) for r in rows])
    if t.size < 3:
        print("  too few frames to judge")
        return 1
    dt = np.diff(t)
    fps_csv = (t.size - 1) / (t[-1] - t[0])
    fps_hdr = hdr[0] if hdr else float("nan")
    med = float(np.median(dt))
    dropped = sum(int(r["dropped_before"]) for r in rows)
    gaps = int((dt > 1.5 * med).sum())
    print(f"  fps: frames.csv {fps_csv:.3f}, header {fps_hdr:.3f} ({100 * (fps_csv / fps_hdr - 1):+.1f} %), "
          f"median interval {med * 1000:.1f} ms")
    print(f"  dropped (writer): {dropped}, gaps > 1.5 intervals: {gaps}, largest gap {dt.max() * 1000:.1f} ms "
          f"({dt.max() / med:.2f} intervals)")
    if abs(fps_csv / fps_hdr - 1) > 0.05:
        fails.append("frames.csv fps differs from the header by more than 5 %")
    if dropped:
        fails.append(f"{dropped} frames dropped by the writer")
    if dt.max() >= 3 * med:
        fails.append("largest gap >= 3 frame intervals")
    ev_path = os.path.join(run_dir, "events.csv")
    if os.path.exists(ev_path):
        ev = list(csv.DictReader(open(ev_path)))
        drives = [(float(a["t_s"]), float(b["t_s"]), a["freq_set_hz"], a.get("v_set_mv", ""))
                  for a, b in zip(ev, ev[1:]) if a["freq_set_hz"] not in ("0", "")]
        for a, b, hz, mv in drives:
            n = int(((t >= a) & (t < b)).sum())
            print(f"  drive {hz} Hz {mv} mV: {b - a:.4f} s, {n} frames ({n / (b - a):.2f} fps)")
    s_path = os.path.join(run_dir, "video_summary.json")
    if os.path.exists(s_path):
        s = json.load(open(s_path))
        print(f"  video_summary: codec {s.get('codec')} q{s.get('quality')}, size {s.get('size')}, "
              f"overlay {s.get('overlay')}")
    print("RESULT:", "OK" if not fails else "FAIL")
    for f in fails:
        print("  FAIL:", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
