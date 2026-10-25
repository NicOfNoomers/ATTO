"""Valveless swing per cycle, rerun B2: short tube, zoomed in, longer 400 and 700 Hz drives (NM, 26 Oct 2026).

Why: in run 20261025_225202 (config B) the swing was clear at 100 and 200 Hz (15-21 nL) but at 400 and 700 Hz
no per-cycle flicker was seen (< 4 nL). The camera exposure is ~4 ms, so a frame averages 1.5 (400 Hz) to 2.7
(700 Hz) cycles and a swing shows mostly as blur. This rerun lowers the floor and tests the line:
- the tube from the pump to the meniscus is shortened (write the length in LINE_MM below, or in the notes);
- the microscope is zoomed in (about 2x: twice the pixels per nL);
- 400 and 700 Hz drive for 30 s (three times the frames), 100, 200 and 1 Hz for 10 s as references.
Same pump (valveless), 2.5 and 1.0 V, two passes (voltage down, then up; frequency order reversed in pass 2).
Operator stops only where the voltage changes (2.5 -> 1.0, then 1.0 -> 2.5 V): the 1.0 V level is run twice in a
row without a stop. At each stop, re-centre the meniscus with room AHEAD of it: at 2.5 V it creeps forward about
0.5 mm per drive (more at 2x zoom on screen) and relaxes back during the rests.
Drives last (N - 0.25)/f (whole strokes, the quarter-period stop), 20 s rests ("rest" blocks).

Recording: 30 fps 1080p MJPG is ~700 MB per minute (~11 GB for this run). Set the dashboard's output folder to
a LOCAL folder (not OneDrive) or pause OneDrive during the run; the 25 Oct run ran out of memory after 5 min.
Bench supply -> driver VM, PPK2 out of the path. Pump ID required, video on, overlay off, codec mjpg.
"""
import json

PORT = "COM3"
LINE_MM = None                      # fill in: tube length from the pump port to the meniscus
TUBE_ID_MM, TUBE_OD_MM = 0.5, 1.5875
LEVELS = [2500, 1000]
PASSES = ["down", "up"]
ORDER = {"down": [100, 700, 200, 400, 1], "up": [1, 400, 200, 700, 100]}
DRIVE_S = {1: 10.0, 100: 10.0, 200: 10.0, 400: 30.0, 700: 30.0}
REST_S = 20.0


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0, "label": "rest"}


def drive(hz, n, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": (n - 0.25) / hz, "repeat": 1, "settle_ms": 0,
            "label": "drive", "strokes": int(n), "edges_commanded": 2 * int(n), **meta}


tests, cur, n_stop, k = [], None, 0, 0
for p, direction in enumerate(PASSES, 1):
    for mv in (LEVELS if direction == "down" else LEVELS[::-1]):
        if mv != cur:
            n_stop += 1
            tests.append({"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0,
                          "note": f"Stop {n_stop} (pass {p}, {direction}), VALVELESS, short tube, zoomed in: set the bench "
                                  f"supply to {mv / 1000:.3f} V, MEASURE VM and type it, re-centre the meniscus with room "
                                  f"ahead of it, then confirm."})
            tests.append(zero(REST_S))
            cur = mv
        for hz in ORDER[direction]:
            k += 1
            tests.append(drive(hz, int(round(DRIVE_S[hz] * hz)), config="B2_valveless", level_mv=mv, reading=k))
            tests[-1]["pass"] = p
            tests.append(zero(REST_S))

cfg = {
    "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Stroke volume vs frequency B2 (valveless, short tube, zoom)",
             "description": ("Swing per cycle vs frequency, WITHOUT valves, rerun with a shorter pump-to-meniscus tube "
                             "and ~2x zoom. 100, 200, 1 Hz for 10 s; 400 and 700 Hz for 30 s; 20 s rests; 2.5 and "
                             "1.0 V; passes down/up; stops only at voltage changes."),
             "operator": "", "pump_id": "", "valves": False, "tube_id_mm": TUBE_ID_MM, "tube_od_mm": TUBE_OD_MM,
             "line_mm": LINE_MM, "zoom": "about 2x vs 25 Oct (scale from the tube OD in the video)"},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": LEVELS[0], "lead_min": 2},
    "power": {"enabled": False},
    "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
}

with open("config_stroke_volume_freq_B2_valveless.json", "w") as f:
    json.dump(cfg, f, indent=2)
nominal = sum(t["duration_s"] for t in tests if t["type"] == "constant")
print(f"config_stroke_volume_freq_B2_valveless.json: {k} drives, {n_stop} voltage stops, nominal {nominal / 60:.1f} min "
      f"(+ stops), video ~{nominal / 60 * 0.7:.0f} GB at 30 fps")
