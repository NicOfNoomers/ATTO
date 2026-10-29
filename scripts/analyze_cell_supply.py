#!/usr/bin/env python3
"""Per-drive-block tables for the coin-cell supply test (make_config_cell_supply.py).

    python scripts/analyze_cell_supply.py <run folder> [<run folder> ...]

For every drive block (from events.csv; t_cmd_s when present) it reads the PPK2 1 ms bins
(power/bins_*.csv[.gz], same clock). V_s per block is the confirmed voltage in force at the
block (events.csv v_set_mv, else voltage_changes.csv, else meta.v_supply_mv).

Table A, measured on the bench:
  peak100   peak current in the first 100 ms (i_max of the bins, i.e. the highest sample)
  plateau   surge plateau current and surge duration (50 ms average stays above the midpoint
            between the early level and the steady level); "-" when there is no surge
  steady    mean current over the last 5 s
  node      bench battery node V_s - I(t) x R_series: minimum from the 1 ms means and the
            worst case from the bins' highest sample; time below 1.8 V
  drop300   largest node drop within any 300 ms window (from 300 ms before the drive)
  charge    charge per block (uAh) and minutes from 12.5 mAh at the block's average current
  With R_series > 0 (v1-v4) the drive is full duty: these node values are NOT the capsule's
  operating point (the coil voltage is the node itself).

Table B, capsule prediction (pump/capsule-driver brief, correction of 29 Oct): with V the coil
voltage during the block (V_s - I x R_series) and I the measured current,
  I_batt = (V / 3.1) x I + 5.3 mA (BLE);  V_node_capsule = 3.1 V - 120 ohm x I_batt;  pass if >= 1.8 V
  for the surge plateau and for steady state, plus minutes from 12.5 mAh at I_batt (steady).

Rows with V_s below the PPK2's 0.8 V ampere-meter rating are flagged "*".
Both tables are also written to <run>/cell_supply_summary.csv.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

CAPSULE = {"v_batt": 3.1, "r_int_ohm": 120.0, "ble_ma": 5.3, "node_min_v": 1.8, "cap_mah": 12.5}
PPK2_MIN_V = 0.8


def load_bins(run):
    files = sorted(glob.glob(os.path.join(run, "power", "bins_*.csv*")))
    if not files:
        raise SystemExit(f"{run}: no power/bins_*.csv - was 'Measure power' on?")
    parts = [np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 3), ndmin=2) for f in files]
    a = np.concatenate([p for p in parts if p.size])
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1], a[:, 2]


def drive_blocks(run, vs_default):
    """[(freq, t_on, t_off, V_s in V)] from events.csv."""
    ev = list(csv.DictReader(open(os.path.join(run, "events.csv"))))
    key = "t_cmd_s" if ev and ev[0].get("t_cmd_s") not in (None, "") else "t_s"
    vc_path = os.path.join(run, "voltage_changes.csv")
    changes = []
    if os.path.exists(vc_path):
        changes = [(float(r["t_s"]), float(r["confirmed_mv"]) / 1000.0)
                   for r in csv.DictReader(open(vc_path)) if r.get("confirmed_mv")]
    out = []
    for a, b in zip(ev, ev[1:]):
        f = float(a["freq_set_hz"])
        if f > 0 and float(b["freq_set_hz"]) == 0:
            t_on, t_off = float(a[key] or a["t_s"]), float(b[key] or b["t_s"])
            if a.get("v_set_mv"):
                vs = float(a["v_set_mv"]) / 1000.0
            else:
                before = [v for t, v in changes if t <= t_on]
                vs = before[-1] if before else vs_default
            out.append((f, t_on, t_off, vs))
    return out


def rolling_max(x, n):
    if x.size < n:
        return np.array([x.max()]) if x.size else x
    return np.lib.stride_tricks.sliding_window_view(x, n).max(axis=1)


def capsule(v_coil, i_ma):
    i_batt = v_coil / CAPSULE["v_batt"] * i_ma + CAPSULE["ble_ma"]
    node = CAPSULE["v_batt"] - CAPSULE["r_int_ohm"] * i_batt / 1000.0
    return i_batt, node


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
    # Capsule prediction: V = coil voltage during the block
    v_coil_st = vs - steady * r / 1000.0
    ib_st, node_cap_st = capsule(v_coil_st, steady)
    if surge_s > 0:
        v_coil_su = vs - plateau * r / 1000.0
        ib_su, node_cap_su = capsule(v_coil_su, plateau)
    else:
        v_coil_su = ib_su = node_cap_su = float("nan")
    nmin = CAPSULE["node_min_v"]
    return {
        "f_hz": f, "v_s": round(vs, 4), "ppk2_below_rating": int(vs < PPK2_MIN_V),
        "t_on_s": round(t_on, 4), "dur_s": round(dur, 3),
        "peak100_ma": round(peak100, 2), "plateau_ma": round(plateau, 2), "surge_s": round(surge_s, 3),
        "steady_ma": round(steady, 3), "mean_ma": round(mean_ma, 3),
        "node_min_v": round(float(node.min()), 4), "node_min_worst_v": round(float(node_worst.min()), 4),
        "below_min_s": round(float((node < vmin_req).sum() * dt), 4),
        "below_min_worst_s": round(float((node_worst < vmin_req).sum() * dt), 4),
        "drop300_v": round(drop300, 4),
        "charge_uah": round(charge_mah * 1000.0, 3),
        "minutes_from_cell": round(cap_mah / mean_ma * 60.0, 1) if mean_ma > 0 else float("inf"),
        "cap_v_coil_surge": round(v_coil_su, 4), "cap_v_coil_steady": round(v_coil_st, 4),
        "cap_i_batt_surge_ma": round(ib_su, 3), "cap_i_batt_steady_ma": round(ib_st, 3),
        "cap_node_surge_v": round(node_cap_su, 4), "cap_node_steady_v": round(node_cap_st, 4),
        "cap_pass_surge": "" if surge_s == 0 else int(node_cap_su >= nmin),
        "cap_pass_steady": int(node_cap_st >= nmin),
        "cap_minutes": round(CAPSULE["cap_mah"] / ib_st * 60.0, 1),
    }


def fmt(x, spec, dash="-"):
    return dash if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def main(runs):
    for run in runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        meta = cfg.get("meta", {})
        r = float(meta.get("series_ohm", 0) or 0)
        vs_meta = meta.get("v_supply_mv", 0)
        vs_default = float(vs_meta[0] if isinstance(vs_meta, list) else vs_meta or 0) / 1000.0
        vmin_req = float(meta.get("node_min_v", 1.8))
        cap = float(meta.get("cell_capacity_mah", 12.5))
        t, i_mean, i_max = load_bins(run)
        rows = []
        for n, (f, t_on, t_off, vs) in enumerate(drive_blocks(run, vs_default), 1):
            res = analyse_block(t, i_mean, i_max, f, t_on, t_off, vs, r, vmin_req, cap)
            if res:
                rows.append({"block": n, **res})
        print(f"\n== {meta.get('variant', '?')} {meta.get('variant_label', '')}  |  R_series {r:g} ohm, "
              f"extra: {meta.get('extra_part', '?')}, pump {meta.get('pump_id') or '?'}")
        print(f"   {os.path.basename(os.path.normpath(run))}")
        if not rows:
            print("   no complete drive block (run stopped or tripped? see power/power_events.csv)")
            continue
        title_a = "A. Measured on the bench" + (" (FULL DUTY behind the resistor: not the capsule operating point)"
                                              if r > 0 else "")
        print(f"   {title_a}")
        print(f"{'#':>3} {'f Hz':>6} {'V_s':>6} {'peak100':>8} {'plateau':>8} {'surge s':>8} {'steady':>7} "
              f"{'node min':>9} {'(worst)':>8} {'<' + format(vmin_req, 'g') + 'V s':>8} {'drop300':>8} "
              f"{'uAh':>7} {'min/cell':>9}")
        for x in rows:
            flag = "*" if x["ppk2_below_rating"] else " "
            # With the resistor shorted the bench node is just V_s: no node columns
            node_cols = (f"{x['node_min_v']:>9.3f} {x['node_min_worst_v']:>8.3f} {x['below_min_s']:>8.3f} "
                         f"{x['drop300_v']:>8.3f}") if r > 0 else f"{'-':>9} {'-':>8} {'-':>8} {'-':>8}"
            print(f"{x['block']:>3} {x['f_hz']:>6g} {x['v_s']:>5.3f}{flag} {x['peak100_ma']:>8.1f} "
                  f"{fmt(x['plateau_ma'], '.1f') if x['surge_s'] else '-':>8} "
                  f"{fmt(x['surge_s'], '.2f') if x['surge_s'] else '-':>8} {x['steady_ma']:>7.2f} "
                  f"{node_cols} {x['charge_uah']:>7.2f} {x['minutes_from_cell']:>9.1f}")
        print(f"   B. Capsule prediction: I_batt = V_coil/{CAPSULE['v_batt']:g} x I + {CAPSULE['ble_ma']:g} mA, "
              f"V_node = {CAPSULE['v_batt']:g} V - {CAPSULE['r_int_ohm']:g} ohm x I_batt, pass >= "
              f"{CAPSULE['node_min_v']:g} V")
        print(f"{'#':>3} {'f Hz':>6} {'V_coil':>7} {'I_surge':>8} {'surge s':>8} {'I_steady':>8} "
              f"{'Ib_surge':>9} {'Ib_steady':>9} {'node_surge':>11} {'node_steady':>12} {'min@Ib':>7}")
        for x in rows:
            flag = "*" if x["ppk2_below_rating"] else " "
            su = x["surge_s"] > 0
            pass_su = "" if not su else (" ok" if x["cap_pass_surge"] else " FAIL")
            pass_st = " ok" if x["cap_pass_steady"] else " FAIL"
            node_su = (fmt(x["cap_node_surge_v"], ".3f") + pass_su) if su else "-"
            print(f"{x['block']:>3} {x['f_hz']:>6g} {x['cap_v_coil_steady']:>6.3f}{flag} "
                  f"{fmt(x['plateau_ma'], '.1f') if su else '-':>8} {fmt(x['surge_s'], '.2f') if su else '-':>8} "
                  f"{x['steady_ma']:>8.2f} {fmt(x['cap_i_batt_surge_ma'], '.2f') if su else '-':>9} "
                  f"{x['cap_i_batt_steady_ma']:>9.2f} {node_su:>11} "
                  f"{format(x['cap_node_steady_v'], '.3f') + pass_st:>12} {x['cap_minutes']:>7.1f}")
        print("   currents in mA, voltages in V; 'worst' uses each 1 ms bin's highest sample;"
              " * = V_s below the PPK2's 0.8 V rating (check against the bench supply's current reading)")
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
