"""Stroke volume against frequency and voltage, microscope on the meniscus (NM via atto-work, 25 Oct 2026).

Why: every 22 Oct ladder was 1 Hz. To compare with Renata's COMSOL model (valveless receiver; outlet volume per
cycle ~6.7 nL at 100 Hz, 2.4 at 200, 0.5 at 400, 0.23 at 700) we need volumes per cycle at operating frequencies.
The camera gives ~15 frames/s, so above ~7.5 Hz single strokes are not resolved:
- Config A, WITH valves: net per stroke from bursts of N strokes, rest to rest (the 22 Oct measure).
- Config B, WITHOUT valves: swing per cycle from the blur/spread band of the meniscus over ~150 frames.

Config A (config_stroke_volume_freq_A_valves.json)
- Frequencies 1 (link to 22 Oct), 100, 200, 400, 700 Hz. Bench-supply levels 2.5, 1.5, 1.0, 0.8 V. 0.8 V
  replaces the proposed 0.6 V, because on 22 Oct 0.6 V at 1 Hz in the 0.5 mm tube gave 0.00, 0.28 and
  -0.19 nL/stroke in three passes: at threshold and not repeatable, so its 1 Hz reference would not move the
  meniscus. 0.8 V still pumped in two of three passes. Every level is also >= 0.8 V, so the PPK2 can stay in
  the supply path if wanted (POWER below).
- N per burst: the meniscus should move about TRAVEL_MM in the 0.5 mm tube (196.35 nL/mm) at the HIGH estimate
  of the net per stroke:
    100-700 Hz, 2.5 V: flow-sensor net with valves, 5 Vpp, primed, open outlet (Stroke_Volume_Diagnosis,
      24 Sep): 15.0-20.6 nL at 100 Hz, 13.5-19.6 at 200-700 Hz; the top of each range is used;
    lower levels: x the 22 Oct 1 Hz ratio net(V)/net(2.5 V) (1 mm run 1, rest to rest) x SAFETY;
    1 Hz: the largest 22 Oct 1 Hz rest-to-rest net at that level, capped at N1_MAX strokes.
  A smaller real net only shortens the travel, and the rest-to-rest resolution is ~1 px = 0.9 nL per burst.
- Burst length (N - 0.25)/f: it stops a quarter period early, so the stop never races the edge of stroke N+1
  (see make_config_stroke_volume.py).
- REPS bursts per (level, frequency) cell, REST_S apart, so each has >= 9 s of rest on both sides
  ([t_on - 9, t_on - 4] and [t_off + 4, t_off + 9]). Before a cell whose travel would push the meniscus out of
  view, the run pauses (a voltage block at the SAME level) so the stage can be re-centred. Then STAGE_SETTLE_S
  of rest follows before the first burst.
- Passes: voltages down with the frequency order 100, 700, 200, 400, 1, then voltages up with the order
  reversed, so frequency, level and time into the run are not confounded.

Config B (config_stroke_volume_freq_B_valveless.json)
- The same frequencies, 10 s of drive each (whole strokes, the same quarter-period stop), 20 s rests, at 2.5
  and 1.0 V; two passes (down, up), with the frequency order alternating as in A.

Both configs: bench supply -> driver VM, power off (POWER = False). Measure VM with the multimeter at every
yellow banner and type THAT value. Dashboard from the stroke-volume branch (perf_counter clock, frames.csv,
t_cmd_s), Pump ID required, video on, overlay off, codec mjpg. Fill in LINE_MM (tube length from the pump port
to the meniscus) before generating: at 100-700 Hz the inertance of that liquid column limits the swing the
meniscus can show (notes: Paper/analysis_2026-10-22_stroke_volume/NOTES_2026-10-25_frequency_configs.md).
"""
import json

PORT = "COM3"
CURRENT_LIMIT_MA = 60
MAX_MV = 2500                       # the pump's limit in the dashboard: set the supply so VM MEASURES <= 2.500 V
POWER = False                       # True: PPK2 in the path (all A levels >= 0.8 V): supply current/energy per burst

TUBE_ID_MM, TUBE_OD_MM = 0.5, 1.5875
LINE_MM = None                      # fill in: tube length from the pump port to the meniscus (for the record)
NL_PER_MM = 3.14159265 * (TUBE_ID_MM / 2) ** 2 * 1000     # 196.35
PX_PER_MM = 221.7                   # 22 Oct microscope setting (OD 352 px = 1/16 in); check it on the day
FREQS = [1, 100, 200, 400, 700]
ORDER = {"down": [100, 700, 200, 400, 1], "up": [1, 400, 200, 700, 100]}

# ---- Config A ----
LEVELS_A = [2500, 1500, 1000, 800]
PASSES_A = ["down", "up"]
REPS = 2
TRAVEL_MM = 2.5                     # target travel per burst at the HIGH net estimate
VIEW_MM = 6.0                       # travel allowed between re-centrings (field of view ~8.7 mm at 221.7 px/mm)
REST_S = 10.0
STAGE_SETTLE_S = 15.0
N1_MAX = 30                         # 1 Hz bursts: at most 30 strokes (as on 22 Oct)
SAFETY = 1.5                        # on the scaled lower-level estimates
Q_HF_2V5 = {100: 20.6, 200: 19.6, 400: 19.6, 700: 19.6}           # nL/stroke, top of the flow-sensor range
Q_1HZ = {2500: 31.86, 1500: 10.57, 1000: 4.77, 800: 2.38, 600: 0.75}   # 22 Oct 1 mm run 1, rest to rest
Q_1HZ_HIGH = {2500: 31.86, 1500: 10.57, 1000: 8.19, 800: 3.74, 600: 0.75}   # largest 22 Oct value per level

# ---- Config B ----
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
    return {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0, "note": text}


def q_high(mv, hz):
    if hz == 1:
        return Q_1HZ_HIGH[mv]
    q = Q_HF_2V5[hz] * Q_1HZ[mv] / Q_1HZ[2500]
    return q if mv == 2500 else q * SAFETY


def strokes_a(mv, hz):
    n = max(1, round(TRAVEL_MM * NL_PER_MM / q_high(mv, hz)))
    return min(n, N1_MAX) if hz == 1 else n


def common(name, desc, levels, extra_meta):
    return {
        "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
        "start_time": None, "end_time": None,
        "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
        "require_pump_id": True,
        "meta": {"name": name, "description": desc, "operator": "", "pump_id": "",
                 "tube_id_mm": TUBE_ID_MM, "tube_od_mm": TUBE_OD_MM, "line_mm": LINE_MM, **extra_meta},
        "serial": {"protocol": "freq_only"},
        "protocol": "freq_only",
        "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
        "supply": {"mode": "manual", "current_limit_ma": CURRENT_LIMIT_MA, "max_mv": MAX_MV,
                   "initial_mv": levels[0], "lead_min": 2},
        "power": {"enabled": bool(POWER)},
        "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
    }


def build_a():
    tests, cells, k, n_pause = [], [], 0, 0
    for p, direction in enumerate(PASSES_A, 1):
        levels = LEVELS_A if direction == "down" else LEVELS_A[::-1]
        for mv in levels:
            travel = VIEW_MM + 1                       # a new level always starts with a pause
            for hz in ORDER[direction]:
                n = strokes_a(mv, hz)
                cell_mm = REPS * n * q_high(mv, hz) / NL_PER_MM
                if travel + cell_mm > VIEW_MM:
                    first = travel > VIEW_MM
                    n_pause += 1
                    what = (f"set the bench supply to {mv / 1000:.3f} V" if first else
                            f"SAME voltage ({mv / 1000:.3f} V): re-type the measured VM")
                    tests.append(pause(mv, f"Pause {n_pause} (pass {p}, {direction}): {what}. Move the stage so the "
                                           f"meniscus has ~{VIEW_MM:.0f} mm of tube in view AHEAD of it (it moves in "
                                           f"the pumping direction), then confirm. Do not touch the stage until the "
                                           f"next yellow banner."))
                    tests.append(zero(STAGE_SETTLE_S))
                    travel = 0.0
                for r in range(1, REPS + 1):
                    k += 1
                    tests.append(drive(hz, n, config="A_valves", pass_=p, level_mv=mv, rep=r, reading=k))
                    tests.append(zero(REST_S))
                travel += cell_mm
                cells.append((p, mv, hz, n, cell_mm / REPS))
    for t in tests:                                    # "pass" is a keyword; keep the 22 Oct key name in the JSON
        if "pass_" in t:
            t["pass"] = t.pop("pass_")
    desc = (f"Net per stroke vs frequency and voltage, WITH valves. {', '.join(map(str, FREQS))} Hz; bench supply "
            f"{', '.join(f'{v / 1000:g}' for v in LEVELS_A)} V; bursts sized for ~{TRAVEL_MM:g} mm meniscus travel in "
            f"the {TUBE_ID_MM} mm tube; {REPS} bursts per cell, {REST_S:.0f} s rests; passes {'/'.join(PASSES_A)}; "
            f"re-centring pauses at the same voltage. Microscope video on the meniscus, frames.csv for timing.")
    cfg = common("Stroke volume vs frequency A (valves)", desc, LEVELS_A,
                 {"valves": True, "reps": REPS, "travel_mm_target": TRAVEL_MM,
                  "cells": [{"pass": p, "level_mv": mv, "hz": hz, "strokes": n} for p, mv, hz, n, _ in cells]})
    cfg["tests"] = tests
    return cfg, cells, n_pause, k


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
            for hz in ORDER[direction]:
                k += 1
                n = int(round(DRIVE_B_S * hz))
                tests.append(drive(hz, n, config="B_valveless", level_mv=mv, reading=k))
                tests[-1]["pass"] = p
                tests.append(zero(REST_B_S))
    desc = (f"Swing per cycle vs frequency, WITHOUT valves. {', '.join(map(str, FREQS))} Hz, {DRIVE_B_S:g} s each, "
            f"{REST_B_S:.0f} s rests, bench supply {', '.join(f'{v / 1000:g}' for v in LEVELS_B)} V, passes "
            f"{'/'.join(PASSES_B)}. Swing read from the blur/spread band of the meniscus (~150 frames per drive).")
    cfg = common("Stroke volume vs frequency B (valveless)", desc, LEVELS_B, {"valves": False})
    cfg["tests"] = tests
    return cfg, n_pause, k


def nominal_s(tests):
    return sum(t["duration_s"] for t in tests if t["type"] == "constant")


if __name__ == "__main__":
    a, cells, pa, ka = build_a()
    b, pb, kb = build_b()
    for fn, cfg in (("config_stroke_volume_freq_A_valves.json", a), ("config_stroke_volume_freq_B_valveless.json", b)):
        with open(fn, "w") as f:
            json.dump(cfg, f, indent=2)
    nl_px = NL_PER_MM / PX_PER_MM
    print(f"config_stroke_volume_freq_A_valves.json: {ka} bursts in {len(cells)} cells, {pa} pauses (voltage set or "
          f"re-centre); nominal {nominal_s(a['tests']) / 60:.1f} min + the pauses (at ~20 s each: "
          f"{(nominal_s(a['tests']) + 20 * pa) / 60:.1f} min)")
    print("  pass level  f(Hz)  N   burst(s)  est. nL/stroke  est. travel mm (px)  +-0.5 ms timing = strokes (%)")
    for p, mv, hz, n, mm in cells:
        if p != 1:
            continue
        dn = 0.5e-3 * hz
        print(f"  {p}  {mv / 1000:4.2f} V {hz:5d} {n:4d} {(n - 0.25) / hz:8.3f}  {q_high(mv, hz):8.2f}       "
              f"{mm:5.2f} ({mm * PX_PER_MM:4.0f})        {dn:5.2f} ({100 * dn / n:4.1f} %)")
    print(f"config_stroke_volume_freq_B_valveless.json: {kb} drives, {pb} pauses; nominal "
          f"{nominal_s(b['tests']) / 60:.1f} min (+ pauses: {(nominal_s(b['tests']) + 20 * pb) / 60:.1f} min)")
    model = {100: 6.7, 200: 2.4, 400: 0.5, 700: 0.23}
    meas = {100: (15.0, 20.6), 200: (13.5, 19.6), 400: (13.5, 19.6), 700: (13.5, 19.6)}
    print(f"  expected swing at 2.5 V in px ({nl_px:.3f} nL/px): model vs our valved net per cycle")
    for hz in (100, 200, 400, 700):
        lo, hi = meas[hz]
        print(f"  {hz:4d} Hz: model {model[hz]:5.2f} nL = {model[hz] / nl_px:5.2f} px; our net {lo:g}-{hi:g} nL = "
              f"{lo / nl_px:4.1f}-{hi / nl_px:4.1f} px")
