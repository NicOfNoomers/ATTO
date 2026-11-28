"""Per-block table for carrier-PWM (duty) runs (capsule-driver note 12): VM, f, carrier duty D, carrier, flow,
current, power, energy per uL, uL per dose; and the test-A equivalence verdict.

Per constant drive block (>= 5 s at f > 0) of merged.csv:
  VM            voltage_changes.csv (confirmed_mv, step function in t_s), else supply.initial_mv
  D, carrier    drive_events.csv (step function in t_s; protocol freq_amp), else 1.0 / -
  flow          mean and SD over the last 60 % (zero = mean of the 0 Hz blocks >= 10 s removed)
  current       PPK2 bins over the same window (PPK2 time -> host time via power/anchors.csv) = battery-side current
  power         VM x I (mW); energy per uL = P / Q (mJ/uL); uL per J
  dose_ul       for blocks <= 15 s: integral of (flow - zero) from the block start to 10 s after it (start transient
                and tail included)
Equivalence (test A): for each f, the duty block at D x VM = V against the D = 1 block at VM = V: flow ratio
(pass if within +-15 %) and I_duty / (D x I_ref) (~1 if the battery current is D x the coil current).
Writes duty_blocks.csv into the run folder.
usage: python scripts/duty_blocks.py <run_dir>
"""
import bisect
import csv
import gzip
import json
import os
import statistics as st
import sys

R = sys.argv[1]
cfg = json.load(open(os.path.join(R, "config.json"), encoding="utf-8"))
vm0 = float((cfg.get("supply") or {}).get("initial_mv") or 2500)


def steps(path, tcol, cols):
    out = []
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            try:
                out.append((float(r[tcol]),) + tuple(r.get(c, "") for c in cols))
            except (ValueError, KeyError):
                pass
    return sorted(out)


vsteps = steps(os.path.join(R, "voltage_changes.csv"), "t_s", ["confirmed_mv"])
dsteps = steps(os.path.join(R, "drive_events.csv"), "t_s", ["carrier_duty", "carrier_hz"])


def at(stp, t, dflt):
    v = dflt
    for row in stp:
        if row[0] <= t + 0.05:
            v = row[1:]
        else:
            break
    return v


t, f, q = [], [], []
for r in csv.DictReader(open(os.path.join(R, "merged.csv"), newline="")):
    try:
        t.append(float(r["t_s"])); f.append(float(r["freq_set_hz"])); q.append(float(r["flow"]))
    except ValueError:
        pass
bt, bi = [], []
pdir = os.path.join(R, "power")
if os.path.isdir(pdir):
    bins = sorted(x for x in os.listdir(pdir) if x.startswith("bins_") and (x.endswith(".csv") or x.endswith(".csv.gz")))
    anc = steps(os.path.join(pdir, "anchors.csv"), "t_sample_s", ["t_host_s"])
    lag = st.median([float(h) - s for s, h in anc]) if anc else 0.0
    if bins:
        b = os.path.join(pdir, next((x for x in bins if x.endswith(".gz")), bins[0]))
        fh = gzip.open(b, "rt", newline="") if b.endswith(".gz") else open(b, newline="")
        for r in csv.DictReader(fh):
            try:
                bt.append(float(r["t_s"]) + lag); bi.append(float(r["i_mean_ma"]))
            except ValueError:
                pass
blocks, i = [], 0
while i < len(t):
    j = i
    while j + 1 < len(t) and f[j + 1] == f[i]:
        j += 1
    blocks.append((t[i], t[j], f[i], i, j))
    i = j + 1
zs = []
for t0, t1, fr, i, j in blocks:
    if fr == 0 and t1 - t0 >= 10:
        zs += [q[k] for k in range(i, j + 1) if k > i + (j - i) * 0.4 and q[k] == q[k]]
z0 = sum(zs) / len(zs) if zs else 0.0
rows = []
for t0, t1, fr, i, j in blocks:
    dur = t1 - t0
    if fr <= 0 or dur < 5:
        continue
    a = t0 + 0.4 * dur
    fl = [q[k] - z0 for k in range(i, j + 1) if t[k] >= a and q[k] == q[k]]
    k0, k1 = bisect.bisect_left(bt, a), bisect.bisect_right(bt, t1)
    cur = sum(bi[k0:k1]) / (k1 - k0) if k1 - k0 > 10 else float("nan")
    vm = float(at(vsteps, t0, (vm0,))[0] or vm0)
    dd = at(dsteps, t0, ("1.0", ""))
    d = float(dd[0] or 1.0)
    car = dd[1]
    qm = sum(fl) / len(fl) if fl else float("nan")
    p_mw = vm / 1000.0 * cur
    dose = ""
    if dur <= 15:
        pts = [(t[k], q[k] - z0) for k in range(i, len(t)) if t0 <= t[k] <= t1 + 10.0 and q[k] == q[k]]
        dose = round(sum(0.5 * (y0 + y1) * (x1 - x0) for (x0, y0), (x1, y1) in zip(pts, pts[1:])) / 60.0, 3)
    rows.append({"t0_s": round(t0, 1), "dur_s": round(dur, 1), "vm_mv": round(vm), "f_hz": int(fr), "carrier_duty": round(d, 4),
                 "carrier_hz": car, "v_eff_mv": round(vm * d), "flow_ul_min": round(qm, 1),
                 "flow_sd": round(st.pstdev(fl), 1) if len(fl) > 1 else "", "i_ma": round(cur, 3),
                 "p_mw": round(p_mw, 2), "mj_per_ul": round(p_mw * 60.0 / qm, 2) if qm and qm > 0.5 else "",
                 "ul_per_j": round(qm / 60.0 / (p_mw / 1000.0), 1) if p_mw > 0 and qm == qm else "", "dose_ul": dose})
with open(os.path.join(R, "duty_blocks.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["t0_s"])
    w.writeheader()
    w.writerows(rows)
print(f"{os.path.basename(R.rstrip('/'))}: zero {z0:+.2f} uL/min, {len(rows)} drive blocks -> duty_blocks.csv")
print(f"{'t0':>7s} {'VM':>5s} {'f':>5s} {'D':>6s} {'carrier':>7s} {'Veff':>5s} {'flow':>7s} {'I mA':>6s} {'P mW':>6s} {'mJ/uL':>6s} {'dose uL':>7s}")
for r in rows:
    print(f"{r['t0_s']:7.1f} {r['vm_mv']:5d} {r['f_hz']:5d} {r['carrier_duty']:6.3f} {str(r['carrier_hz']):>7s} {r['v_eff_mv']:5d} "
          f"{r['flow_ul_min']:7.1f} {r['i_ma']:6.2f} {r['p_mw']:6.1f} {str(r['mj_per_ul']):>6s} {str(r['dose_ul']):>7s}")
ref = {(r["f_hz"], r["v_eff_mv"]): r for r in rows if r["carrier_duty"] >= 0.999}
pairs = [(r, ref.get((r["f_hz"], r["v_eff_mv"]))) for r in rows if r["carrier_duty"] < 0.999]
pairs = [(dr, rr) for dr, rr in pairs if rr]
if pairs:
    print("\nequivalence (duty vs D = 1 at the same effective voltage):")
    for dr, rr in pairs:
        ratio = dr["flow_ul_min"] / rr["flow_ul_min"] if rr["flow_ul_min"] else float("nan")
        irat = dr["i_ma"] / (dr["carrier_duty"] * rr["i_ma"]) if rr["i_ma"] else float("nan")
        print(f"   {dr['f_hz']:5d} Hz, {dr['v_eff_mv']} mV eff, D {dr['carrier_duty']:.2f} @ {dr['carrier_hz']} Hz: flow "
              f"{dr['flow_ul_min']:.1f} vs {rr['flow_ul_min']:.1f} ({100 * (ratio - 1):+.0f} %, "
              f"{'PASS' if abs(ratio - 1) <= 0.15 else 'FAIL'}); I_duty / (D x I_ref) {irat:.2f}")
