#!/usr/bin/env python3
"""Per-drive-block table for the coin-cell supply test (make_config_cell_supply.py).

    python scripts/analyze_cell_supply.py <run folder> [<run folder> ...]

For every drive block (from events.csv; t_cmd_s when present) it reads the PPK2 1 ms bins
(power/bins_*.csv[.gz], same clock) and reports:
  peak100   peak current in the first 100 ms (i_max of the bins, i.e. the highest sample)
  plateau   surge plateau current and surge duration (50 ms average stays above the midpoint
            between the early level and the steady level); "-" when there is no surge
  steady    mean current over the last 5 s
  node      battery node V_node(t) = V_s - I(t) x R_series: minimum from the 1 ms means and the
            worst case from the bins' highest sample; time below node_min_v (1.8 V)
  drop300   largest node drop within any 300 ms window (from 300 ms before the drive)
  charge    charge per block (uAh) and the minutes of pumping 12.5 mAh would give at the
            block's average current
V_s is the voltage confirmed in voltage_changes.csv (the measured value), else meta.v_supply_mv.
One table per run (variant); also written to <run>/cell_supply_summary.csv.
"""
import csv
import glob
import json
import os
import sys

import numpy as np


def load_bins(run):
    files = sorted(glob.glob(os.path.join(run, "power", "bins_*.csv*")))
    if not files:
        raise SystemExit(f"{run}: no power/bins_*.csv - was 'Measure power' on?")
    parts = [np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 3), ndmin=2) for f in files]
    a = np.concatenate([p for p in parts if p.size])
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1], a[:, 2]


def drive_blocks(run):
    ev = list(csv.DictReader(open(os.path.join(run, "events.csv"))))
    key = "t_cmd_s" if ev and ev[0].get("t_cmd_s") not in (None, "") else "t_s"
    out = []
    for a, b in zip(ev, ev[1:]):
        f = float(a["freq_set_hz"])
        if f > 0 and float(b["freq_set_hz"]) == 0:
            out.append((f, float(a[key] or a["t_s"]), float(b[key] or b["t_s"])))
    return out


def rolling_max(x, n):
    if x.size < n:
        return np.array([x.max()]) if x.size else x
    return np.lib.stride_tricks.sliding_window_view(x, n).max(axis=1)


def analyse_block(t, i_mean, i_max, f, t_on, t_off, vs, r, vmin_req, cap_mah):
    sel = (t >= t_on) & (t < t_off)
    if sel.sum() < 50:
        return None
    tt, I, Imax = t[sel] - t_on, i_mean[sel], i_max[sel]
    dt = float(np.median(np.diff(t[sel])))
    dur = t_off - t_on
    peak100 = float(Imax[tt < 0.1].max()) if (tt < 0.1).any() else float("nan")
    steady = float(I[tt >= dur - 5.0].mean())
    k = max(1, int(round(0.05 / dt)))
    m = np.convolve(I, np.ones(k) / k, mode="same")
    early_sel = (tt >= 0.1) & (tt < 0.4)
    early = float(np.median(m[early_sel])) if early_sel.any() else steady
    surge_s, plateau = 0.0, float("nan")
    # Behind 120 ohm the surge is small (~25 % above steady), so the test is relative + absolute
    if early - steady > max(1.0, 0.08 * steady):
        thr = 0.5 * (early + steady)
        below = np.flatnonzero((tt >= 0.1) & (m < thr))
        surge_s = float(tt[below[0]]) if below.size else float(dur)
        psel = (tt >= 0.1) & (tt < max(0.15, surge_s - 0.05))
        plateau = float(np.median(I[psel])) if psel.any() else early
    node = vs - I * r / 1000.0
    node_worst = vs - Imax * r / 1000.0
    pre = (t >= t_on - 0.3) & (t < t_off)
    v_pre = vs - i_mean[pre] * r / 1000.0
    n300 = max(2, int(round(0.3 / dt)))
    drop300 = float((rolling_max(v_pre, n300) - v_pre[n300 - 1:]).max()) if v_pre.size >= n300 else 0.0
    charge_mah = float(I.sum() * dt / 3600.0)
    mean_ma = float(I.mean())
    return {
        "f_hz": f, "t_on_s": round(t_on, 4), "dur_s": round(dur, 3),
        "peak100_ma": round(peak100, 2), "plateau_ma": round(plateau, 2), "surge_s": round(surge_s, 3),
        "steady_ma": round(steady, 3), "mean_ma": round(mean_ma, 3),
        "node_min_v": round(float(node.min()), 4), "node_min_worst_v": round(float(node_worst.min()), 4),
        "below_min_s": round(float((node < vmin_req).sum() * dt), 4),
        "below_min_worst_s": round(float((node_worst < vmin_req).sum() * dt), 4),
        "drop300_v": round(drop300, 4),
        "charge_uah": round(charge_mah * 1000.0, 3),
        "minutes_from_cell": round(cap_mah / mean_ma * 60.0, 1) if mean_ma > 0 else float("inf"),
    }


def main(runs):
    for run in runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        meta = cfg.get("meta", {})
        r = float(meta.get("series_ohm", 0) or 0)
        vs = float(meta.get("v_supply_mv", 0) or 0) / 1000.0
        vc_path = os.path.join(run, "voltage_changes.csv")
        if os.path.exists(vc_path):
            vc = list(csv.DictReader(open(vc_path)))
            if vc and vc[0].get("confirmed_mv"):
                vs = float(vc[0]["confirmed_mv"]) / 1000.0
        vmin_req = float(meta.get("node_min_v", 1.8))
        cap = float(meta.get("cell_capacity_mah", 12.5))
        t, i_mean, i_max = load_bins(run)
        rows = []
        for n, (f, t_on, t_off) in enumerate(drive_blocks(run), 1):
            res = analyse_block(t, i_mean, i_max, f, t_on, t_off, vs, r, vmin_req, cap)
            if res:
                rows.append({"block": n, **res})
        print(f"\n== {meta.get('variant', '?')} {meta.get('variant_label', '')}  |  V_s {vs:.3f} V, "
              f"R_series {r:g} ohm, extra: {meta.get('extra_part', '?')}, pump {meta.get('pump_id') or '?'}")
        print(f"   {os.path.basename(os.path.normpath(run))}")
        hdr = (f"{'#':>3} {'f Hz':>6} {'peak100':>8} {'plateau':>8} {'surge s':>8} {'steady':>7} "
               f"{'node min':>9} {'(worst)':>8} {'<' + format(vmin_req, 'g') + 'V s':>8} {'drop300':>8} "
               f"{'uAh':>7} {'min/cell':>9}")
        print(hdr)
        for x in rows:
            plateau = "-" if x["surge_s"] == 0 else f"{x['plateau_ma']:.1f}"
            surge = "-" if x["surge_s"] == 0 else f"{x['surge_s']:.2f}"
            print(f"{x['block']:>3} {x['f_hz']:>6g} {x['peak100_ma']:>8.1f} {plateau:>8} {surge:>8} "
                  f"{x['steady_ma']:>7.2f} {x['node_min_v']:>9.3f} {x['node_min_worst_v']:>8.3f} "
                  f"{x['below_min_s']:>8.3f} {x['drop300_v']:>8.3f} {x['charge_uah']:>7.2f} "
                  f"{x['minutes_from_cell']:>9.1f}")
        if not rows:
            print("   no complete drive block (run stopped or tripped? see power/power_events.csv)")
        print("   currents in mA; node in V; 'worst' uses each 1 ms bin's highest sample")
        if rows:
            out = os.path.join(run, "cell_supply_summary.csv")
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
