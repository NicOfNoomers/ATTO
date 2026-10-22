#!/usr/bin/env python3
"""Rebuild per-frame times from the burned-in overlay of old-format videos.

Runs recorded before the frames.csv video log (video_recording_partNNNN.mp4,
header says 30 fps, real rate ~15 fps) carry their timing only in the overlay:
"Time: 123.45s" (run time on the same clock as events.csv, taken right after
the camera returned the frame) and "Freq: 1 Hz" (the drive setpoint then).
This script reads both from every frame by re-rendering whole overlay lines
with OpenCV's Hershey font and keeping the best match (Dice overlap):
  * time: candidates from the previous frame's time + 0.00..0.50 s (0.01 s
    steps), widened if nothing matches well; the first frame is searched over
    the whole run,
  * frequency: the setpoints that occur in events.csv.
Output: frames_from_overlay.csv in the run folder:

    frame_index, video_file, video_frame, t_s, freq_hz, dice_t, dice_f, freq_events_hz

dice_t / dice_f are the match scores (~0.9 is a clean read; check rows below
0.8). freq_events_hz is the setpoint from events.csv at t_s, a cross-check: it
should equal freq_hz except within one frame interval of a command.

The overlay was drawn with OpenCV 4's putText; OpenCV 5 renders text
differently, so run this with OpenCV 4 (pip install "opencv-python<5").

Usage: python scripts/overlay_to_frames.py <run folder>
Timing of these runs: the overlay time came from time.monotonic on Python 3.10
(15.6 ms ticks), shown with 2 decimals.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2      # as VideoRecorder.add_overlay drew it
TIME_ORG, FREQ_ORG = (10, 30), (10, 55)
TIME_WIN = (slice(8, 36), slice(5, 210))
FREQ_WIN = (slice(33, 61), slice(5, 160))


def render(text, org, win):
    img = np.zeros((100, 260), np.uint8)
    cv2.putText(img, text, org, FONT, SCALE, 255, THICK)
    return img[win] > 127


def mask_green(roi):
    b, g, r = (roi[:, :, k].astype(np.float32) for k in range(3))
    return (g - np.maximum(r, b)) > 110


def mask_yellow(roi):
    b, g, r = (roi[:, :, k].astype(np.float32) for k in range(3))
    return (np.minimum(g, r) - b) > 110


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else 0.0


_cache = {}


def time_template(cs):
    """Rendered 'Time: x.xxs' for a value in hundredths of a second."""
    t = _cache.get(cs)
    if t is None:
        t = _cache[cs] = render(f"Time: {cs / 100:.2f}s", TIME_ORG, TIME_WIN)
        if len(_cache) > 20000:
            _cache.clear()
    return t


def best_time(m, lo_cs, hi_cs):
    best, best_d = None, -1.0
    for cs in range(max(0, lo_cs), hi_cs + 1):
        d = dice(time_template(cs), m)
        if d > best_d:
            best, best_d = cs, d
    return best, best_d


def main(run_dir):
    if int(cv2.__version__.split(".")[0]) >= 5:
        print(f"WARNING: OpenCV {cv2.__version__}: the overlay was drawn with OpenCV 4's font; "
              f"matches will be poor. Use OpenCV 4 (pip install \"opencv-python<5\").")
    files = sorted(glob.glob(os.path.join(run_dir, "video_recording_part*.mp4")))
    if not files:
        sys.exit("no video_recording_part*.mp4 in " + run_dir)
    ev = list(csv.DictReader(open(os.path.join(run_dir, "events.csv"))))
    ev_t = np.array([float(e["t_s"]) for e in ev])
    ev_f = [int(float(e["freq_set_hz"])) for e in ev]
    freqs = sorted(set(ev_f))
    freq_tpl = {hz: render(f"Freq: {hz} Hz", FREQ_ORG, FREQ_WIN) for hz in freqs}
    t_max_cs = int((ev_t.max() + 120) * 100)
    out_path = os.path.join(run_dir, "frames_from_overlay.csv")
    rows, idx, prev = [], 0, None
    for f in files:
        cap = cv2.VideoCapture(f)
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            roi = frame[:100, :260]
            mt, mf = mask_green(roi)[TIME_WIN], mask_yellow(roi)[FREQ_WIN]
            if prev is None:
                cs, d = best_time(mt, 0, t_max_cs)
            else:
                cs, d = best_time(mt, prev, prev + 50)
                if d < 0.8:                       # dropped frames or a bad read: widen
                    cs2, d2 = best_time(mt, prev, prev + 1000)
                    if d2 > d:
                        cs, d = cs2, d2
            prev = cs if d >= 0.8 else prev if prev is not None else cs
            hz, dfq = max(((h, dice(tpl, mf)) for h, tpl in freq_tpl.items()), key=lambda x: x[1])
            t = cs / 100.0
            j = int(np.searchsorted(ev_t, t, side="right")) - 1
            rows.append([idx, os.path.basename(f), k, f"{t:.2f}", hz, f"{d:.3f}", f"{dfq:.3f}",
                         ev_f[j] if j >= 0 else ""])
            idx += 1
            k += 1
        cap.release()
        print(f"  {os.path.basename(f)}: {k} frames, t {rows[-k][3]} .. {rows[-1][3]} s", flush=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["frame_index", "video_file", "video_frame", "t_s", "freq_hz", "dice_t", "dice_f",
                    "freq_events_hz"])
        w.writerows(rows)

    t = np.array([float(r[3]) for r in rows])
    dt = np.diff(t)
    weak = sum(1 for r in rows if float(r[5]) < 0.8)
    nonincr = int((dt <= 0).sum())
    mism = sum(1 for r in rows if r[7] != "" and r[4] != r[7])
    print(f"{out_path}\n  frames {len(rows)}; weak time reads (dice < 0.8): {weak}; non-increasing times: {nonincr}")
    print(f"  median interval {1000 * np.median(dt):.1f} ms ({1 / np.median(dt):.2f} fps), "
          f"largest gap {1000 * dt.max():.0f} ms")
    print(f"  overlay Freq vs events.csv: {mism} frames differ (expected only right at commands)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
