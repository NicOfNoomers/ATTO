"""Free flow per drive block of a run folder, for the bubble-flush loop (numpy only, runs on the lab PC).

flow = mean of merged.csv flow over the last 60 % of the drive block, minus the mean over the last 60 % of the
0 Hz blocks right before and after (offset). Current = PPK2 bins over the same window, when present.
Prints one row per drive block, then for a scan (config_scan_quick.json) the peak frequency and the flow at 600 Hz:

    py -3.10 scripts\\flow_per_block.py pump_runs\\<run folder> [<run folder> ...]
    py -3.10 scripts\\flow_per_block.py pump_runs\\<scan run> --pick      # last line: PICK <f> (the peak frequency)
"""
import csv
import glob
import gzip
import json
import os
import sys

import numpy as np


def read_csv_cols(path, cols, opener=open):
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        data = {c: [] for c in cols}
        for row in r:
            for c in cols:
                try:
                    data[c].append(float(row[c]))
                except (TypeError, ValueError):
                    data[c].append(np.nan)
    return {c: np.array(v) for c, v in data.items()}


def blocks(rd):
    tests = json.load(open(os.path.join(rd, "config.json"), encoding="utf-8"))["tests"]
    te = read_csv_cols(os.path.join(rd, "events.csv"), ["t_s"])["t_s"]
    n = [2 if x["type"] == "voltage" else 1 for x in tests]     # a voltage step writes two event rows
    starts = np.cumsum([0] + n)
    nb = int(np.searchsorted(starts, len(te), side="right") - 1)
    return [(tests[k], te[starts[k]], te[starts[k + 1]] if starts[k + 1] < len(te) else te[-1]) for k in range(nb)]


def bins_files(rd):
    """The PPK2 bins files of a run, each hour once: on the share FolderSync leaves the .csv it copied while the
    run was writing it next to the final .csv.gz (it never deletes), so prefer the .gz when both exist."""
    files = sorted(glob.glob(os.path.join(rd, "power", "bins_*.csv*")))
    gz = {f[:-3] for f in files if f.endswith(".csv.gz")}
    return [f for f in files if not (f.endswith(".csv") and f in gz)]


def power(rd):
    t, i = [], []
    for f in bins_files(rd):
        d = read_csv_cols(f, ["t_s", "i_mean_ma"], gzip.open if f.endswith(".gz") else open)
        t.append(d["t_s"]); i.append(d["i_mean_ma"])
    return (np.concatenate(t), np.concatenate(i)) if t else (None, None)


def win_mean(t0, t1, tt, yy):
    if tt is None:
        return np.nan
    m = (tt >= t0 + 0.4 * (t1 - t0)) & (tt < t1)
    return float(np.nanmean(yy[m])) if m.any() else np.nan


def analyse(rd):
    bl = blocks(rd)
    mg = read_csv_cols(os.path.join(rd, "merged.csv"), ["t_s", "flow"])
    tp, ip = power(rd)
    rows = []
    for k, (t, t0, t1) in enumerate(bl):
        if t["type"] != "constant" or not t.get("frequency_hz"):
            continue
        z = [win_mean(bl[j][1], bl[j][2], mg["t_s"], mg["flow"]) for j in (k - 1, k + 1)
             if 0 <= j < len(bl) and bl[j][0]["type"] == "constant" and not bl[j][0].get("frequency_hz")]
        base = float(np.nanmean(z)) if z else 0.0
        rows.append({"k": k, "t_min": t0 / 60, "f": int(t["frequency_hz"]), "dur": t["duration_s"],
                     "q": win_mean(t0, t1, mg["t_s"], mg["flow"]) - base, "i_ma": win_mean(t0, t1, tp, ip)})
    return rows


def main(argv):
    pick = "--pick" in argv
    for rd in [a for a in argv if not a.startswith("--")]:
        rows = analyse(rd)
        print(f"\n{os.path.basename(os.path.normpath(rd))}")
        print(f"{'k':>4} {'t_min':>6} {'f_Hz':>6} {'dur_s':>6} {'flow_uL_min':>12} {'I_mA':>6}")
        for r in rows:
            print(f"{r['k']:>4} {r['t_min']:>6.2f} {r['f']:>6} {r['dur']:>6.0f} {r['q']:>12.1f} {r['i_ma']:>6.1f}")
        if rows:
            best = max(rows, key=lambda r: -1e9 if np.isnan(r["q"]) else r["q"])
            q600 = [r["q"] for r in rows if r["f"] == 600]
            print(f"peak {best['f']} Hz {best['q']:.1f} uL/min; 600 Hz {q600[-1]:.1f} uL/min" if q600 else
                  f"peak {best['f']} Hz {best['q']:.1f} uL/min")
            if pick:
                print(f"PICK {best['f']}")


if __name__ == "__main__":
    main(sys.argv[1:])
