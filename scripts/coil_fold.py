"""Fold XIAO coil-current captures at the drive period and pair them with the run's state (NM, 27 Nov 2026).

For each capture in <cap_dir>/captures.csv: run time = capture wall time - run start (meta.json 'created'), drive
frequency at that time and the block's flow (merged.csv), the current folded over one FULL drive period into 64
phase bins (the coil current is bidirectional, period 1/f), offset removed with the mean of captures taken at 0 Hz.
Features: mean |i|, RMS, peak-to-peak, |H2|/|H1|, |H3|/|H1| of the folded waveform; waveforms saved for overlays.
Writes <cap_dir>/fold.csv and fold_waves.csv; with --plot, an overlay per frequency coloured by flow.
usage: python scripts/coil_fold.py <cap_dir> <run_dir> [--plot fig.png]
"""
import argparse
import bisect
import csv
import datetime
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cap_dir")
ap.add_argument("run_dir")
ap.add_argument("--plot", default=None)
a = ap.parse_args()
NB = 64
meta = json.load(open(os.path.join(a.run_dir, "meta.json"), encoding="utf-8"))
created = meta.get("created") or meta.get("created_at")
t_run0 = datetime.datetime.fromisoformat(str(created)).timestamp()
t, f, q = [], [], []
for r in csv.DictReader(open(os.path.join(a.run_dir, "merged.csv"), newline="")):
    try:
        t.append(float(r["t_s"])); f.append(float(r["freq_set_hz"])); q.append(float(r["flow"]))
    except (ValueError, KeyError):
        pass
caps = list(csv.DictReader(open(os.path.join(a.cap_dir, "captures.csv"))))
recs = []
for c in caps:
    n, fs, lsb = int(c["n"]), float(c["fs_hz"]), float(c["lsb_ma"])
    cur = np.fromfile(os.path.join(a.cap_dir, f"cap_{int(c['k']):05d}.bin"), dtype="<i2").astype(float) * lsb
    tr = float(c["t_unix"]) - t_run0
    t1 = tr + n / fs
    i0, i1 = bisect.bisect_left(t, tr - 0.2), bisect.bisect_right(t, t1 + 0.2)
    fset = sorted(set(f[i0:i1]))
    j0, j1 = bisect.bisect_left(t, tr - 5), bisect.bisect_right(t, t1 + 5)
    fr = fset[0] if len(fset) == 1 else None
    fl = [q[k] for k in range(j0, j1) if fr is not None and f[k] == fr and q[k] == q[k]]
    recs.append({"k": int(c["k"]), "t_run": tr, "f": fr, "flow": (sum(fl) / len(fl)) if fl else None, "cur": cur, "fs": fs})
zeros = [r["cur"].mean() for r in recs if r["f"] == 0]
off = float(np.mean(zeros)) if zeros else 0.0
print(f"{len(recs)} captures; offset from {len(zeros)} captures at 0 Hz: {off:+.3f} mA")
rows, waves = [], []
for r in recs:
    if not r["f"]:
        continue
    cur = r["cur"] - off
    per = r["fs"] / r["f"]
    ph = (np.arange(len(cur)) % per) / per
    idx = np.minimum((ph * NB).astype(int), NB - 1)
    wf = np.bincount(idx, weights=cur, minlength=NB) / np.maximum(np.bincount(idx, minlength=NB), 1)
    sp = np.abs(np.fft.rfft(wf))
    rows.append({"k": r["k"], "t_run_s": round(r["t_run"], 2), "f": int(r["f"]),
                 "flow": "" if r["flow"] is None else round(r["flow"], 1),
                 "mean_abs_ma": round(float(np.mean(np.abs(cur))), 3), "rms_ma": round(float(np.sqrt(np.mean(cur ** 2))), 3),
                 "pp_ma": round(float(wf.max() - wf.min()), 3),
                 "h2": round(float(sp[2] / sp[1]), 4) if sp[1] > 0 else "", "h3": round(float(sp[3] / sp[1]), 4) if sp[1] > 0 else ""})
    waves.append([r["k"], int(r["f"]), "" if r["flow"] is None else round(r["flow"], 1)] + [round(float(x), 4) for x in wf])
with open(os.path.join(a.cap_dir, "fold.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["k", "t_run_s", "f", "flow", "mean_abs_ma", "rms_ma", "pp_ma", "h2", "h3"])
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(a.cap_dir, "fold_waves.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["k", "f", "flow"] + [f"b{i}" for i in range(NB)])
    w.writerows(waves)
print(f"folded {len(rows)} captures during a drive -> fold.csv, fold_waves.csv")
if a.plot and waves:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freqs = sorted({wv[1] for wv in waves})
    fig, axes = plt.subplots(len(freqs), 1, figsize=(6.4, 2.2 * len(freqs)), squeeze=False)
    for ax, fr in zip(axes[:, 0], freqs):
        sel = [wv for wv in waves if wv[1] == fr]
        fls = [wv[2] for wv in sel if wv[2] != ""]
        lo, hi = (min(fls), max(fls)) if fls else (0, 1)
        for wv in sel:
            c = plt.cm.viridis((wv[2] - lo) / (hi - lo + 1e-9)) if wv[2] != "" else "0.5"
            ax.plot(np.arange(NB) / NB, wv[3:], color=c, lw=0.8)
        ax.set_title(f"{fr} Hz (colour: flow {lo:.0f}-{hi:.0f} uL/min)", fontsize=8)
        ax.set_ylabel("coil current (mA)")
    axes[-1, 0].set_xlabel("phase (drive periods)")
    fig.tight_layout()
    fig.savefig(a.plot, dpi=150)
    print(f"plot: {a.plot}")
