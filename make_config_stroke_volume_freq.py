"""Stroke volume against frequency and voltage, microscope on the meniscus (NM via atto-work, 25 Oct 2026;
Config A redone for the programmable supply, 2 Nov 2026).

Config A, WITH valves (config_stroke_volume_freq_A_valves_<tube>mm.json, one file per tube ID)
Why (2 Nov): the 1-2 Nov low-f run on pump 261030-1 (O:/All/nicmi/atto_lab_pc/20261101_233208_409754_Low_frequency_
vs_voltage; Paper/analysis_2026-11-02_lowf_voltage_261030-1/README.md) measured ~270 nL per cycle at 1-2 Hz with the
flow sensor, falling with frequency as V(f) = V0 tanh(1/(4 f tau)); the pump is a ~90 ohm resistor at 1-100 Hz. The
meniscus under the microscope checks that volume independently of the flow sensor, and the supply's current
readback takes the measurement below 0.8 V, where the PPK2 is not verified.
- Frequencies 1, 2, 5, 10, 20 Hz (the time-constant regime) and 100 Hz (link to the earlier runs).
- Levels 2.5, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4 V. 0.8 and 1.0 V overlap the PPK2's verified range, to re-check the
  supply-readback calibration on the day (1-2 Nov: supply reading = 0.980 x I_PPK2 - 3.2 mA, whole mA, +-0.5 mA,
  reads 0 at 0.3 V). Blocks below 0.8 V carry "ppk2_below_range": true.
- Voltage steps by the RS PRO RSD3305P (supply.mode "rsd3305p"): the supply sets each level by itself. A voltage
  block has "pause": true only where the stage must be re-centred (the run then waits for Done - continue).
- Power ON at every level (PPK2 in line, 10 ms bins) plus supply/supply.csv (1 s polls). Current does not depend
  on frequency at 1-100 Hz, so the drive seconds of all bursts at a level are pooled: each level gets at least
  CURRENT_MIN_S of drive over the run (the 1 Hz bursts are lengthened where the travel-sized bursts fall short).
- N per burst: the meniscus moves about TRAVEL_MM at the HIGH estimate V0 tanh(1/(4 f tau)) of the level
  (Q_FIT); below 0.8 V the stroke is unknown (extrapolation ~160/130/105 nL at 0.6/0.5/0.4 V), so the 0.8 V
  values are the upper bound: nothing overruns the view, a smaller real stroke only shortens the travel.
- Burst length (N - 0.25)/f: it stops a quarter period early, so the stop never races the edge of stroke N+1.
- REPS bursts per (level, frequency), REST_S apart, so each has >= 9 s of rest on both sides; the first burst
  after a level change also has REST_S of rest after the step. Before a burst whose travel would push the
  meniscus out of the VIEW_MM ahead of it, the run pauses (voltage block at the SAME level, "pause": true) to
  re-centre the stage; STAGE_SETTLE_S of rest follows.
- Passes: voltages down with the frequency order ORDER_A["down"], then up with the order reversed.
- No priming or conditioning inside the run: 600 Hz for 60 s would push ~200 uL, ~1 m of meniscus travel in
  the 0.5 mm tube. Prime BEFORE the tube is aligned under the microscope.
- Tube: 0.5 mm ID = 196 nL/mm (1.4 mm per stroke at 2.5 V, 1 Hz: few strokes per view, many pauses);
  1.0 mm ID = 785 nL/mm (0.34 mm per stroke, ~3.5 nL per px). Load the file of the tube that is fitted.

Config B, WITHOUT valves (config_stroke_volume_freq_B_valveless.json, unchanged since 25 Oct)
- 1, 100, 200, 400, 700 Hz, 10 s of drive each (whole strokes, the same quarter-period stop), 20 s rests, at 2.5
  and 1.0 V; two passes (down, up); manual bench supply, power off. Swing per cycle from the blur/spread band.
  Fill in LINE_MM (tube length from the pump port to the meniscus) before generating.

usage: python make_config_stroke_volume_freq.py [--address COMx | host:5025] [--tube-id 0.5 1.0]
  Without --address the A configs carry none and the dashboard uses the Power panel's Address field.
"""
import argparse
import json
import math

ap = argparse.ArgumentParser(description="Stroke volume vs frequency configs A (valves) and B (valveless)")
ap.add_argument("--address", default=None, help="RSD3305P address for config A: COMx, or host:5025 (supply_bridge.py)")
ap.add_argument("--tube-id", type=float, nargs="+", default=[0.5, 1.0], help="observation tube ID(s) in mm for A")
args = ap.parse_args()

PORT = "COM3"
CURRENT_LIMIT_MA = 60
MAX_MV = 2500                       # the pump's limit: the supply refuses anything above it
LINE_MM = None                      # fill in: tube length from the pump port to the meniscus (for the record)
PX_PER_MM = 221.7                   # 22 Oct microscope setting (OD 352 px = 1/16 in); check it on the day
TUBE_OD_MM = {0.5: 1.5875}          # 1/16 in; fill in the 1.0 mm tube's OD if known


def nl_per_mm(tube_id):
    return math.pi * (tube_id / 2) ** 2 * 1000


# ---- Config A (programmable supply) ----
LEVELS_A = [2500, 1500, 1000, 800, 600, 500, 400]
PPK2_MIN_MV = 800                   # PPK2 ammeter verified from 0.8 V (ppk2-api clamps its calibration there)
FREQS_A = [1, 2, 5, 10, 20, 100]
ORDER_A = {"down": [100, 2, 20, 1, 10, 5], "up": [5, 10, 1, 20, 2, 100]}
PASSES_A = ["down", "up"]
REPS = 2
TRAVEL_MM = 2.5                     # target travel per burst at the HIGH stroke estimate
VIEW_MM = 6.0                       # travel allowed between re-centrings (field of view ~8.7 mm at 221.7 px/mm)
REST_S = 10.0
STAGE_SETTLE_S = 15.0
STEP_SETTLE_S = 2.0                 # after an automatic voltage step, before the REST_S zero
CURRENT_MIN_S = 20.0                # pooled drive seconds per level over the run (1 s supply polls)
PAUSE_GUESS_S = 20.0                # operator time per re-centring, for the run-length estimate
# 1-2 Nov fit, pump 261030-1: V0 (nL per cycle), tau (s) per level; below 0.8 V the 0.8 V values (upper bound)
Q_FIT = {2500: (268.0, 0.030), 2000: (260.0, 0.031), 1500: (266.0, 0.036), 1000: (237.0, 0.038), 800: (211.0, 0.041)}
Q_EXTRAP_1HZ = {600: 160.0, 500: 130.0, 400: 105.0}     # nL, extrapolated (for the record only)
TRIP_MA, HOLD_MS = 57, 300

# ---- Config B (unchanged) ----
TUBE_ID_B = 0.5
FREQS_B = [1, 100, 200, 400, 700]
ORDER_B = {"down": [100, 700, 200, 400, 1], "up": [1, 400, 200, 700, 100]}
LEVELS_B = [2500, 1000]
PASSES_B = ["down", "up"]
DRIVE_B_S = 10.0
REST_B_S = 20.0


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, n, **meta):
    # strokes / edges_commanded and the rest are for the analysis (the runner ignores them)
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": (n - 0.25) / hz, "repeat": 1,
            "settle_ms": 0, "strokes": int(n), "edges_commanded": 2 * int(n), **meta}


def pause(mv, text):
    """Config B: manual bench supply stop."""
    return {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0, "note": text}


def q_high(mv, hz):
    v0, tau = Q_FIT[max(mv, PPK2_MIN_MV)]
    return v0 * math.tanh(1.0 / (4.0 * hz * tau))


def common(name, desc, tube_id, extra_meta, supply, power):
    return {
        "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
        "start_time": None, "end_time": None,
        "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
        "require_pump_id": True,
        "meta": {"name": name, "description": desc, "operator": "", "pump_id": "",
                 "tube_id_mm": tube_id, "tube_od_mm": TUBE_OD_MM.get(tube_id), "line_mm": LINE_MM, **extra_meta},
        "serial": {"protocol": "freq_only"},
        "protocol": "freq_only",
        "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
        "supply": supply,
        "power": power,
        "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
    }


def strokes_a(tube_id):
    """N per (level, f): ~TRAVEL_MM at the high estimate, then the 1 Hz (then 2 Hz) bursts lengthened until each
    level has CURRENT_MIN_S of pooled drive over the run, never past VIEW_MM per burst."""
    npm = nl_per_mm(tube_id)
    n = {(mv, hz): max(1, round(TRAVEL_MM * npm / q_high(mv, hz))) for mv in LEVELS_A for hz in FREQS_A}
    visits = len(PASSES_A) * REPS
    for mv in LEVELS_A:
        def pooled():
            return visits * sum((n[mv, hz] - 0.25) / hz for hz in FREQS_A)
        for hz in (1, 2):
            while pooled() < CURRENT_MIN_S and (n[mv, hz] + 1) * q_high(mv, hz) / npm <= VIEW_MM:
                n[mv, hz] += 1
    return n


def build_a(tube_id):
    npm = nl_per_mm(tube_id)
    n_of = strokes_a(tube_id)
    tests, cells, k, n_pause = [], [], 0, 0
    travel = VIEW_MM + 1                               # the run starts with the stage being set
    note_recentre = (f"move the stage so the meniscus has ~{VIEW_MM:.0f} mm of tube in view AHEAD of it (it moves in "
                     f"the pumping direction), then press Done - continue. Do not touch the stage until the next "
                     f"banner.")

    def flag(mv):
        return {"ppk2_below_range": True} if mv < PPK2_MIN_MV else {}

    for p, direction in enumerate(PASSES_A, 1):
        levels = LEVELS_A if direction == "down" else LEVELS_A[::-1]
        for mv in levels:
            first_mm = n_of[mv, ORDER_A[direction][0]] * q_high(mv, ORDER_A[direction][0]) / npm
            step = {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": STEP_SETTLE_S,
                    "level": f"pass {p} {direction}", **flag(mv)}
            if travel + first_mm > VIEW_MM:
                n_pause += 1
                step.update(pause=True, note=f"Pause {n_pause} (pass {p}, {mv / 1000:.2f} V): {note_recentre}")
                tests += [step, zero(STAGE_SETTLE_S)]
                travel = 0.0
            else:
                step["note"] = f"Pass {p}: {mv / 1000:.2f} V, set by the supply"
                tests += [step, zero(REST_S)]
            for hz in ORDER_A[direction]:
                n = n_of[mv, hz]
                burst_mm = n * q_high(mv, hz) / npm
                for r in range(1, REPS + 1):
                    if travel + burst_mm > VIEW_MM:
                        n_pause += 1
                        tests.append({"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0,
                                      "pause": True, **flag(mv),
                                      "note": f"Pause {n_pause} (pass {p}, SAME {mv / 1000:.2f} V): {note_recentre}"})
                        tests.append(zero(STAGE_SETTLE_S))
                        travel = 0.0
                    k += 1
                    tests.append(drive(hz, n, config="A_valves", pass_=p, level_mv=mv, rep=r, reading=k,
                                       q_high_nl=round(q_high(mv, hz), 1), **flag(mv)))
                    tests.append(zero(REST_S))
                    travel += burst_mm
                cells.append((p, mv, hz, n, burst_mm))
    for t in tests:                                    # "pass" is a keyword; keep the 22 Oct key name in the JSON
        if "pass_" in t:
            t["pass"] = t.pop("pass_")
    pooled = {mv: len(PASSES_A) * REPS * sum((n_of[mv, hz] - 0.25) / hz for hz in FREQS_A) for mv in LEVELS_A}
    desc = (f"Net per stroke vs frequency and voltage, WITH valves, {tube_id} mm ID observation tube. "
            f"{', '.join(map(str, FREQS_A))} Hz; levels {', '.join(f'{v / 1000:g}' for v in LEVELS_A)} V set by the "
            f"RSD3305P programmable supply (no operator steps; yellow banners only to re-centre the stage). Bursts "
            f"sized for ~{TRAVEL_MM:g} mm meniscus travel at V0 tanh(1/(4 f tau)) from the 1-2 Nov run (0.8 V values "
            f"below 0.8 V); {REPS} bursts per cell, {REST_S:.0f} s rests; passes {'/'.join(PASSES_A)}. POWER ON (PPK2 "
            f"in line; below 0.8 V the PPK2 is unverified, cross-check with supply/supply.csv). Prime BEFORE aligning "
            f"the tube: no priming in this run. Tick Record video; type the Pump ID.")
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": CURRENT_LIMIT_MA, "max_mv": MAX_MV,
              "initial_mv": LEVELS_A[0], "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
    if args.address:
        supply["address"] = args.address
    power = {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": TRIP_MA,
             "watch_hold_ms": HOLD_MS, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0}
    cfg = common(f"Stroke volume vs frequency A (valves, {tube_id} mm tube, supply)", desc, tube_id,
                 {"valves": True, "reps": REPS, "travel_mm_target": TRAVEL_MM, "view_mm": VIEW_MM,
                  "nl_per_mm": round(npm, 2), "px_per_mm": PX_PER_MM, "ppk2_min_mv": PPK2_MIN_MV,
                  "q_fit_nl_tau_s": {str(k_): list(v) for k_, v in Q_FIT.items()},
                  "q_extrap_1hz_nl": {str(k_): v for k_, v in Q_EXTRAP_1HZ.items()},
                  "supply_readback_cal": "supply mA = 0.980 x I_PPK2 - 3.2 mA (1-2 Nov), whole mA, reads 0 at 0.3 V",
                  "pooled_drive_s": {str(mv): round(s, 1) for mv, s in pooled.items()},
                  "cells": [{"pass": p, "level_mv": mv, "hz": hz, "strokes": n} for p, mv, hz, n, _ in cells]},
                 supply, power)
    cfg["tests"] = tests
    return cfg, cells, n_pause, k, pooled


def build_b():
    tests, n_pause, k = [], 0, 0
    for p, direction in enumerate(PASSES_B, 1):
        levels = LEVELS_B if direction == "down" else LEVELS_B[::-1]
        for mv in levels:
            n_pause += 1
            tests.append(pause(mv, f"Pause {n_pause} (pass {p}, {direction}), VALVELESS assembly: set the bench "
                                   f"supply to {mv / 1000:.3f} V, MEASURE VM and type it, centre the meniscus, "
                                   f"then confirm."))
            tests.append(zero(REST_B_S))
            for hz in ORDER_B[direction]:
                k += 1
                n = int(round(DRIVE_B_S * hz))
                tests.append(drive(hz, n, config="B_valveless", level_mv=mv, reading=k))
                tests[-1]["pass"] = p
                tests.append(zero(REST_B_S))
    desc = (f"Swing per cycle vs frequency, WITHOUT valves. {', '.join(map(str, FREQS_B))} Hz, {DRIVE_B_S:g} s each, "
            f"{REST_B_S:.0f} s rests, bench supply {', '.join(f'{v / 1000:g}' for v in LEVELS_B)} V, passes "
            f"{'/'.join(PASSES_B)}. Swing read from the blur/spread band of the meniscus (~150 frames per drive).")
    supply = {"mode": "manual", "current_limit_ma": CURRENT_LIMIT_MA, "max_mv": MAX_MV,
              "initial_mv": LEVELS_B[0], "lead_min": 2}
    cfg = common("Stroke volume vs frequency B (valveless)", desc, TUBE_ID_B, {"valves": False}, supply,
                 {"enabled": False})
    cfg["tests"] = tests
    return cfg, n_pause, k


def nominal_s(tests):
    return sum(t["duration_s"] for t in tests if t["type"] == "constant") + \
        sum(t.get("settle_s", 0) for t in tests if t["type"] == "voltage")


def tube_tag(tube_id):
    return f"{tube_id:.1f}mm"


if __name__ == "__main__":
    for tube_id in args.tube_id:
        a, cells, pa, ka, pooled = build_a(tube_id)
        fn = f"config_stroke_volume_freq_A_valves_{tube_tag(tube_id)}.json"
        with open(fn, "w") as f:
            json.dump(a, f, indent=2)
        npm = nl_per_mm(tube_id)
        n_steps = sum(1 for t in a["tests"] if t["type"] == "voltage")
        nom = nominal_s(a["tests"])
        total = nom + n_steps * 1.0 + pa * PAUSE_GUESS_S + 10.0
        print(f"{fn}: {ka} bursts, {len(cells)} cells, {pa} re-centring pauses, {n_steps} voltage blocks; "
              f"nominal {nom / 60:.1f} min, ~{total / 60:.0f} min with ~1 s per step and {PAUSE_GUESS_S:.0f} s per "
              f"pause; supply address {args.address or 'NOT SET (Power panel Address field)'}")
        print(f"  {npm:.1f} nL/mm, {npm / PX_PER_MM:.2f} nL/px at {PX_PER_MM} px/mm. Pass 1:")
        print("  level   f(Hz)    N  burst(s)  high nL/stroke  travel mm (px)")
        for p, mv, hz, n, mm in cells:
            if p == 1:
                print(f"  {mv / 1000:4.2f} V {hz:5d} {n:5d} {(n - 0.25) / hz:8.3f}  {q_high(mv, hz):9.1f}       "
                      f"{mm:5.2f} ({mm * PX_PER_MM:4.0f})" + ("   PPK2 below range" if mv < PPK2_MIN_MV else ""))
        print("  pooled drive per level over the run (s): " +
              ", ".join(f"{mv / 1000:g} V {s:.0f}" for mv, s in pooled.items()))
    b, pb, kb = build_b()
    with open("config_stroke_volume_freq_B_valveless.json", "w") as f:
        json.dump(b, f, indent=2)
    print(f"config_stroke_volume_freq_B_valveless.json: {kb} drives, {pb} pauses; nominal "
          f"{nominal_s(b['tests']) / 60:.1f} min (+ pauses: {(nominal_s(b['tests']) + 20 * pb) / 60:.1f} min)")
