"""Measure today's volume per pulse from a finished run, and optionally write a
hold config whose pulse rates hit given target flows.

    python calibrate_pulses.py <run_folder>
    python calibrate_pulses.py <run_folder> --targets 600:3,600:30,100:3 --hold-min 20

Works on any run whose config contains pulse groups ([ON at f Hz, OFF] x N), e.g.
config_session_calibration.json. Needs numpy only (no pandas).

Why: on the October pump a 0.1 s pulse delivers a fixed volume whatever the
rest after it (periods 1-30 s), but that volume moves between sessions
(14 Oct: 464 nL at 600 Hz; 15 Oct: ~545). Holds set from yesterday's volume
landed -11 to +18 % off target while staying steady within 3-4 %. Measuring
the volume at the start of the session removes most of that offset.

Method: events.csv has one row per command of the expanded config, so every
command maps to its block by position. Per pulse, flow is integrated from one
ON command to the next and the baseline (linear between the zero block before
and the zero block after the group, each over its trailing part) subtracted.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def n_cmds(x):
    if x["type"] == "constant":
        return 1
    if x["type"] == "sweep":
        return x["steps"] * x["repeat"]
    return x["repeat"] * sum(n_cmds(b) for b in x["blocks"])


def load(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    with open(os.path.join(run, "events.csv"), newline="") as fh:
        te = np.array([float(r["t_s"]) for r in csv.DictReader(fh)])
    m = np.genfromtxt(os.path.join(run, "merged.csv"), delimiter=",", names=True)
    return cfg, te, m["t_s"].astype(float), m["flow"].astype(float)


def analyse(run):
    cfg, te, t, q = load(run)
    tests = cfg["tests"]
    starts = np.cumsum([0] + [n_cmds(x) for x in tests])
    if starts[-1] != len(te):
        sys.exit(f"command log has {len(te)} rows, config expands to {starts[-1]}: "
                 "run incomplete or config changed -- cannot map blocks")
    T0 = te[starts[:-1]]
    T1 = np.append(te[starts[1:-1]], t[-1])

    def anchor(k):
        x = tests[k]
        if not (x["type"] == "constant" and x["frequency_hz"] == 0):
            return None
        keep = min(max(x["duration_s"] - 8.0, 5.0), 40.0)       # skip the settling tail
        a = max(T0[k], T1[k] - keep)
        sel = (t >= a) & (t < T1[k])
        return 0.5 * (a + T1[k]), float(q[sel].mean())

    groups, warn = [], []
    for k, x in enumerate(tests):
        if x["type"] != "group" or x["blocks"][0]["frequency_hz"] <= 0:
            continue
        before = anchor(k - 1) if k > 0 else None
        after = anchor(k + 1) if k + 1 < len(tests) else None
        if before is None or after is None:
            warn.append(f"block {k}: no zero block on both sides, skipped")
            continue
        (tb, zb), (ta, za) = before, after
        if abs(za - zb) > 0.3:
            warn.append(f"block {k}: baseline moved {za - zb:+.2f} uL/min across the group")
        a, b = starts[k], starts[k + 1]
        on_t = te[a:b:2]
        ends = np.append(on_t[1:], T1[k])
        vols = []
        for s0, s1 in zip(on_t, ends):
            sel = (t >= s0) & (t < s1)
            tt, qq = t[sel], q[sel]
            if len(tt) < 2:
                continue
            base = zb + (za - zb) * (tt - tb) / (ta - tb)
            dt = np.diff(tt, append=s1)
            vols.append(np.sum((qq - base) * dt) / 60.0 * 1e3)       # nL
        v = np.array(vols)
        groups.append(dict(k=k, hz=x["blocks"][0]["frequency_hz"],
                           on=x["blocks"][0]["duration_s"],
                           period=x["blocks"][0]["duration_s"] + x["blocks"][1]["duration_s"],
                           n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else math.nan,
                           t_min=T0[k] / 60))
    return groups, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--targets", default="", help="e.g. 600:3,600:30,100:3 (Hz:uL/min)")
    ap.add_argument("--hold-min", type=float, default=20.0)
    ap.add_argument("--out", default="config_calibrated_holds.json")
    a = ap.parse_args()

    groups, warn = analyse(a.run)
    if not groups:
        sys.exit("no pulse groups found")
    print(f"run: {os.path.basename(os.path.normpath(a.run))}")
    print("  block   Hz   ON s  period s  pulses   nL/pulse    sd    t (min)")
    for g in groups:
        print(f"  {g['k']:5d} {g['hz']:4d}  {g['on']:5.2f}  {g['period']:8.2f}  {g['n']:6d}  "
              f"{g['mean']:9.1f}  {g['sd']:6.1f}  {g['t_min']:7.1f}")
    cal = {}
    print("\nper frequency (all groups pooled, pulse-weighted):")
    for hz in sorted({g["hz"] for g in groups}):
        gs = [g for g in groups if g["hz"] == hz]
        n = sum(g["n"] for g in gs)
        mean = sum(g["mean"] * g["n"] for g in gs) / n
        sd = math.sqrt(sum((g["n"] - 1) * g["sd"] ** 2 for g in gs if g["n"] > 1) / max(n - len(gs), 1))
        cal[hz] = (mean, gs[0]["on"])
        print(f"  {hz:5d} Hz: {mean:7.1f} nL per {gs[0]['on']:.2f} s pulse  (pulse sd {sd:.0f} nL, "
              f"CV {100*sd/mean:.0f} %, n {n}, standard error {100*sd/mean/math.sqrt(n):.1f} %)")
    for w in warn:
        print("  WARNING:", w)

    if not a.targets:
        return
    holds = []
    print("\npulse period for each target:")
    for item in a.targets.split(","):
        hz, flow = item.split(":")
        hz, flow = int(hz), float(flow)
        if hz not in cal:
            sys.exit(f"{hz} Hz was not calibrated in this run")
        v_nl, on = cal[hz]
        per = 60.0 * v_nl * 1e-3 / flow
        note = ""
        if per < 3 * on:
            note = "  <- rest shorter than 2x pulse: pulses deliver less, flow will undershoot"
        if per > 60:
            note = "  <- fewer than 1 pulse per minute: flow is lumpy"
        print(f"  {hz:5d} Hz @ {flow:7.2f} uL/min -> one {on:.2f} s pulse every {per:7.2f} s{note}")
        holds.append((hz, flow, on, round(per, 2)))

    zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                      "repeat": 1, "settle_ms": 0}
    sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                        "repeat": 1, "settle_ms": 0}
    burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
             "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}
    seg_s = 300.0
    nseg = max(1, int(round(a.hold_min * 60 / seg_s)))
    tests = [zero(60.0), dict(burst), zero(30.0)]
    for hz, flow, on, per in holds:
        tests.append(zero(60.0))
        for s in range(nseg):
            n = max(1, math.ceil(seg_s / per))
            tests += [{"type": "group", "repeat": n,
                       "blocks": [sp(hz, on), zero(round(per - on, 2))]},
                      zero(20.0 if s < nseg - 1 else 60.0)]
    tests += [zero(20.0), dict(burst), zero(60.0)]
    src = json.load(open(os.path.join(a.run, "config.json")))
    cfg = {k: src[k] for k in ("port", "baud", "read_timeout_s", "start_time", "end_time",
                               "break_seconds_between_tests", "send_zero_on_exit",
                               "serial", "flow_sensor", "protocol") if k in src}
    cfg["tests"] = tests
    cfg["meta"] = {"name": "Calibrated holds",
                   "description": (f"Holds set from pulse volumes measured in "
                                   f"{os.path.basename(os.path.normpath(a.run))}: "
                                   + ", ".join(f"{hz} Hz {cal[hz][0]:.0f} nL" for hz in cal)
                                   + ". Targets: " + ", ".join(f"{h} Hz @ {f:g}" for h, f, _, _ in holds)
                                   + f" uL/min, {nseg} x 5 min segments each."),
                   "operator": "", "pump_number": src.get("meta", {}).get("pump_number", "")}
    json.dump(cfg, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}: {len(holds)} holds x {nseg} x 5 min")


if __name__ == "__main__":
    main()
