"""Smallest stroke volume, precision run: fine 1 Hz voltage ladder around the threshold, repeated 3 times.

Why (NM, 22 Oct 2026): the first ladder (run 20261022_132158) gave a net forward volume of ~1.4-2.9 nL
per stroke at 0.6 V and none at 0.4 V and below, so the threshold lies between 0.4 and 0.6 V. This run
resolves it with a thinner observation tube (more meniscus travel per nL) and checks repeatability:
- 50 mV steps from 0.4 to 0.7 V, plus 0.8 and 1.0 V (clearly pumping) and 0.3 V (clearly not);
- three passes, DOWN, UP, DOWN, so time into the run and voltage are no longer confounded and every
  level is visited 3 times (the turning-point levels twice in a row);
- 30 strokes per step (more strokes -> a tighter per-stroke slope; 20 gave SE ~0.05 nL);
- 40 s rest before each drive: the meniscus drifts back for ~1-2 min after a drive, and the analysis fits
  that drift over the rest before the next drive.

Per step: voltage block (0 Hz until the bench supply is set and the MEASURED value confirmed) -> rest
REST_S -> drive (STROKES - 0.25) / HZ s at HZ (exactly 2 x STROKES bridge edges, see
make_config_stroke_volume.py). The first rest is FIRST_REST_S, the last rest REST_S.

Measure the voltage at the driver's VM pin with the multimeter and type THAT value in the confirmation:
near the threshold 50 mV matters, and a bench supply's display is not that accurate at 0.3-0.7 V.

Wiring as for config_stroke_volume.json: bench supply -> driver VM directly, PPK2 out of the path,
power off. Dashboard: Pump ID (required), video recording on, overlay off, codec mjpg.
"""
import json

HZ = 1                                       # integer: freq_only sends whole Hz
STROKES = 30
REST_S = 40.0
FIRST_REST_S = 60.0
LEVELS_MV = [1000, 800, 700, 650, 600, 550, 500, 450, 400, 300]     # high -> low
PASSES = ["down", "up", "down"]
CURRENT_LIMIT_MA = 60
PORT = "COM3"
TUBE_ID_MM = None                            # fill in: the new, thinner observation tube (for the record)
TUBE_OD_MM = None

assert float(HZ).is_integer(), "freq_only sends whole Hz"
drive_s = (STROKES - 0.25) / HZ


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(s, mv, p, k):
    return {"type": "constant", "frequency_hz": int(HZ), "duration_s": float(s), "repeat": 1, "settle_ms": 0,
            "strokes": STROKES, "edges_commanded": 2 * STROKES, "pass": p, "level_mv": mv, "reading": k}


sequence = []
for p, direction in enumerate(PASSES, 1):
    lv = LEVELS_MV if direction == "down" else LEVELS_MV[::-1]
    sequence += [(p, mv) for mv in lv]

tests = []
for k, (p, mv) in enumerate(sequence, 1):
    tests.append({"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0,
                  "note": f"Reading {k}/{len(sequence)} (pass {p}): set the bench supply to {mv / 1000:.3f} V, "
                          f"MEASURE VM at the driver and type the measured value, keep the meniscus in view, "
                          f"then confirm."})
    tests += [zero(FIRST_REST_S if k == 1 else REST_S), drive(drive_s, mv, p, k)]
tests.append(zero(REST_S))

cfg = {
    "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {
        "name": "Stroke volume low-voltage ladder x3",
        "description": (f"Smallest stroke volume, precision + repeatability. {HZ} Hz, {STROKES} strokes per step, "
                        f"bench-supply levels {', '.join(f'{v / 1000:g}' for v in LEVELS_MV)} V, passes "
                        f"{'/'.join(PASSES)} ({len(sequence)} readings, operator-confirmed measured voltage), "
                        f"{REST_S:.0f} s rest before each drive. Thinner observation tube. Microscope video on "
                        f"the meniscus, frames.csv for timing. PPK2 out of the supply path."),
        "operator": "", "pump_id": "",
        "tube_id_mm": TUBE_ID_MM, "tube_od_mm": TUBE_OD_MM,
        "sequence_mv": [mv for _, mv in sequence],
    },
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": {"mode": "manual", "current_limit_ma": CURRENT_LIMIT_MA, "max_mv": 2500,
               "initial_mv": sequence[0][1], "lead_min": 2},
    "power": {"enabled": False},
    "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
}

with open("config_stroke_volume_low.json", "w") as f:
    json.dump(cfg, f, indent=2)

nominal = FIRST_REST_S + (len(sequence) - 1) * REST_S + len(sequence) * drive_s + REST_S
print("config_stroke_volume_low.json")
print(f"  levels {', '.join(f'{v / 1000:g}' for v in LEVELS_MV)} V; passes {'/'.join(PASSES)} -> {len(sequence)} readings")
print(f"  each: confirm measured voltage -> {REST_S:.0f} s rest -> {drive_s:g} s at {HZ} Hz ({STROKES} strokes)")
print(f"  run time {nominal:.0f} s = {nominal / 60:.1f} min + your {len(sequence)} confirmations")
# expected meniscus motion, from run 20261022_132158 (1.0 mm tube; nL per stroke while driving)
net = {1000: 7.0, 800: 4.1, 700: 2.7, 650: 2.0, 600: 1.4, 550: 0.8, 500: 0.3, 450: 0.0, 400: 0.0, 300: 0.0}
swing = {1000: 301, 800: 279, 700: 255, 650: 240, 600: 228, 550: 205, 500: 185, 450: 165, 400: 148, 300: 84}
for idm in (0.5, 0.3):
    nl_mm = 3.14159 * (idm / 2) ** 2 * 1000
    per_pass = sum(net[v] for v in LEVELS_MV) * STROKES / nl_mm
    print(f"  {idm} mm ID ({nl_mm:.0f} nL/mm): swing up to {max(swing.values()) / nl_mm:.1f} mm p-p, "
          f"1.4 nL = {1.4 / nl_mm * 1000:.0f} um; net travel <= {per_pass:.1f} mm per pass "
          f"(less after the back-drift)")
