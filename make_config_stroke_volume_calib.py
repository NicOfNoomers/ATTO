"""Calibration run before the valved stroke-volume run: volume per stroke of THIS pump in THIS tube, per voltage and
frequency, with small bursts that stay inside the microscope field (NM, 26 Oct 2026).

Why: config A sized its bursts from guesses (flow-sensor nets and the 22 Oct 1 Hz voltage ratios). In the quick test
20261026_020015 (pump 260930-4, 0.8 mm bore tube) the 1 Hz and 1.5 V bursts moved 6-10x more than planned (a 15-stroke
1 Hz burst at 2.5 V moved ~4.9 uL, half the field) while the 400/700 Hz bursts moved 1-2 px. The meniscus left the
field after ~200 s. This run measures q (nL per stroke) per cell; the main run is then sized from it so every burst
moves the same ~30-40 px.

Design
- Levels ASCENDING 0.8, 1.0, 1.5, 2.5 V (small volumes first); per level the frequencies 700, 400, 200, 100, 1 Hz
  (smallest expected volume first).
- Two bursts per cell: N1 sized for ~TARGET_PX of travel at the HIGH estimate of q, N2 = 4 x N1 (8 x at 400/700 Hz,
  where the quick test gave only ~1 nL/stroke against a high estimate of 15), so the second burst is still measurable
  when q is well below the high estimate. Bursts last (N - 0.25)/f; REST_S of rest before and after each burst.
- Re-centring: an operator stop (voltage block at the SAME level) is inserted whenever the HIGH-estimate travel since
  the last stop would exceed BUDGET_PX. At every stop put the meniscus near the LEFT edge of the image (it moves right).
  Stops also happen at each voltage change.
- High estimates of q (nL/stroke) from the quick test and the 25 Oct runs 220735/223420 (rounded up):
  2.5 and 1.5 V: 1 Hz 400, 100 Hz 60, 200 Hz 60, 400/700 Hz 15;  1.0 and 0.8 V: 1 Hz 200, 100/200 Hz 30, 400/700 Hz 10.
- Scale used for sizing only: tube 1/16 in OD = 43.5 px (27.4 px/mm), 0.8 mm bore -> 18.3 nL per px (TUBE_ID_MM: confirm).
- Supply: run UNclipped (NM, 26 Oct). The 25-26 Oct valved runs were current-limited. At 2.5 V the 10 ms mean current
  is ~83 mA in the first ~1.5 s of a drive, with 0.3-0.9 A spikes at the stroke edges (PPK2, 100 kS/s). The config
  sets 600 mA, as the valveless runs (PPK2 trip at 95 % = 570 mA, on the 10 ms mean). Every burst here is shorter than
  the high-current phase at 2.5 V, as in config A.
PPK2 in the supply path (power on), video on (overlay off, mjpg), Pump ID required.
"""
import json
import math

PORT = "COM3"
CURRENT_LIMIT_MA = 600                         # as the valveless runs (NM, 26 Oct): no clipping
TUBE_OD_MM, TUBE_ID_MM = 1.5875, 0.8          # NM: new tube, 0.8 mm bore (OD assumed 1/16 in)
PX_PER_MM = 43.5 / TUBE_OD_MM                 # quick test 20261026_020015
NL_PER_PX = math.pi * (TUBE_ID_MM / 2) ** 2 * 1000 / PX_PER_MM
TARGET_PX = 15.0                               # first burst at the high estimate; the second is 4x
BUDGET_PX = 450.0                              # high-estimate travel allowed between re-centring stops (field 640 px)
REST_S = 10.0
LEVELS = [800, 1000, 1500, 2500]
FREQS = [700, 400, 200, 100, 1]
Q_HIGH = {2500: {1: 400, 100: 60, 200: 60, 400: 15, 700: 15},
          1500: {1: 400, 100: 60, 200: 60, 400: 15, 700: 15},
          1000: {1: 200, 100: 30, 200: 30, 400: 10, 700: 10},
          800: {1: 200, 100: 30, 200: 30, 400: 10, 700: 10}}


def zero(s, label="rest"):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0, "label": label}


def drive(hz, n, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": (n - 0.25) / hz, "repeat": 1, "settle_ms": 0,
            "label": "burst", "strokes": int(n), "edges_commanded": 2 * int(n), **meta}


def stop(mv, k, why):
    return {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0, "label": "stop",
            "note": f"Stop {k}: {why} Put the meniscus near the LEFT edge of the image (it moves right), then confirm."}


tests, cells, n_stop, budget, k = [], [], 0, 0.0, 0
for mv in LEVELS:
    n_stop += 1
    tests.append(stop(mv, n_stop, f"set the bench supply to {mv / 1000:.3f} V (limit {CURRENT_LIMIT_MA} mA), MEASURE VM and type it."))
    tests.append(zero(REST_S, "still"))
    budget = 0.0
    for hz in FREQS:
        q = Q_HIGH[mv][hz]
        n1 = max(2, round(TARGET_PX * NL_PER_PX / q))
        n2 = (8 if hz >= 400 else 4) * n1      # 400/700 Hz may pump ~1 nL/stroke (quick test): a larger second burst
        cell_px = (n1 + n2) * q / NL_PER_PX
        if budget + cell_px > BUDGET_PX:
            n_stop += 1
            tests.append(stop(mv, n_stop, f"re-centre, SAME voltage ({mv / 1000:.3f} V): re-type the measured VM."))
            tests.append(zero(REST_S, "still"))
            budget = 0.0
        for rep, n in ((1, n1), (2, n2)):
            k += 1
            tests.append(drive(hz, n, config="calibration", level_mv=mv, rep=rep, reading=k))
            tests.append(zero(REST_S))
        budget += cell_px
        cells.append((mv, hz, n1, n2, cell_px))

cfg = {
    "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Stroke volume calibration (valves)",
             "description": ("Volume per stroke per voltage and frequency, small bursts that stay in the microscope field: "
                             "0.8/1.0/1.5/2.5 V ascending, 700/400/200/100/1 Hz, two bursts per cell (N and 4N, 8N at 400/700 Hz; N sized for "
                             f"~{TARGET_PX:.0f} px at the high q estimate), {REST_S:.0f} s rests, re-centring stops when the "
                             "high-estimate travel reaches the field. Sizes the main valved run."),
             "operator": "", "pump_id": "", "valves": True, "tube_id_mm": TUBE_ID_MM, "tube_od_mm": TUBE_OD_MM,
             "line_mm": None, "sizing": {"px_per_mm": round(PX_PER_MM, 2), "nl_per_px": round(NL_PER_PX, 2),
                                          "target_px": TARGET_PX, "budget_px": BUDGET_PX, "q_high": Q_HIGH},
             "cells": [{"level_mv": mv, "hz": hz, "n1": n1, "n2": n2} for mv, hz, n1, n2, _ in cells]},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": {"mode": "manual", "current_limit_ma": CURRENT_LIMIT_MA, "max_mv": 2500, "initial_mv": LEVELS[0], "lead_min": 2},
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0},
    "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
}

with open("config_stroke_volume_calib.json", "w") as f:
    json.dump(cfg, f, indent=2)
nominal = sum(t["duration_s"] for t in tests if t["type"] == "constant")
print(f"config_stroke_volume_calib.json: {k} bursts in {len(cells)} cells, {n_stop} stops "
      f"({n_stop - len(LEVELS)} re-centring); nominal {nominal / 60:.1f} min + stops")
print(f"  sizing: {PX_PER_MM:.1f} px/mm, {NL_PER_PX:.1f} nL/px (0.8 mm bore)")
print("  level  f(Hz)  N1   N2   burst2(s)  high-estimate travel per cell (px)")
for mv, hz, n1, n2, px in cells:
    print(f"  {mv / 1000:4.2f} V {hz:5d} {n1:4d} {n2:5d} {(n2 - 0.25) / hz:8.3f}   {px:5.0f}")
