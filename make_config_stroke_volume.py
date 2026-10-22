"""Smallest stroke volume: a 1 Hz voltage ladder filmed under the microscope.

Goal (NM, 22 Oct 2026): find the smallest volume the pump can push. At 1 Hz
the bench-supply voltage is stepped down from 2.5 V to 0.1 V, 20 strokes per
step. From the video the analysis measures the net meniscus advance per
stroke (and the per-stroke oscillation); the smallest volume is the advance at
the lowest voltage where it is still clearly above zero.

Per voltage step:
    voltage block   the run pauses at 0 Hz until the bench supply is set and
                    the value typed in and confirmed (yellow banner)
    rest  20 s      0 Hz, drift reference before the drive
    drive 19.75 s   1 Hz = 20 strokes (see below)
then a final 20 s rest.

Drive length: (strokes - 0.25) / f, not strokes / f. A square wave started at
the command flips every 1/(2f); after exactly strokes/f the stop command and
the edge that would start stroke 21 are due at the same instant, so a
fraction of a millisecond decides between 20 and 20.5 strokes. Stopping a
quarter period early (mid half-cycle) gives exactly 2 x strokes edges with a
+-0.25/f margin (+-250 ms at 1 Hz). The last half-cycle is 0.25 s instead
of 0.5 s; the coil current settles in ~1.5 ms, so the stroke is complete.
How the Arduino firmware starts and stops a cycle is not known yet: check the
first drive's stroke count in the video. Every frame's time is in frames.csv on the same clock
as events.csv; voltage_changes.csv has the confirmed voltages.

Wiring: bench supply -> driver VM directly. The PPK2 is NOT in the path: in
ampere-meter mode it needs 0.8-5.0 V and this ladder goes down to 0.1 V, so
power measurement is off. The bench supply's current limit is the protection.

Dashboard: type the Pump ID (required), tick "Enable video recording", leave
"Overlay" off (the sync comes from frames.csv), codec mjpg.
"""
import json

HZ = 1                                   # integer: the freq_only protocol sends whole Hz
STROKES = 20
REST_S = 20.0
VOLTAGES_MV = [2500, 2000, 1500, 1000, 800, 600, 400, 300, 200, 100]
CURRENT_LIMIT_MA = 60                    # bench supply limit (pump driver slides: 60 mA)
PORT = "COM3"                            # the rig's Arduino

assert float(HZ).is_integer(), "freq_only sends whole Hz"
drive_s = (STROKES - 0.25) / HZ           # stop mid half-cycle: unambiguous stroke count


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(s):
    # strokes / edges_commanded are for the analysis (the runner ignores them):
    # 2 x strokes bridge edges, the first at the 1 Hz command (events.csv t_cmd_s)
    return {"type": "constant", "frequency_hz": int(HZ), "duration_s": float(s), "repeat": 1, "settle_ms": 0,
            "strokes": STROKES, "edges_commanded": 2 * STROKES}


tests = []
for i, mv in enumerate(VOLTAGES_MV, 1):
    tests.append({"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": 0,
                  "note": f"Step {i}/{len(VOLTAGES_MV)}: set the bench supply to {mv / 1000:.3f} V, keep the "
                          f"meniscus in view (move the stage), then confirm."})
    tests += [zero(REST_S), drive(drive_s)]
tests.append(zero(REST_S))

cfg = {
    "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {
        "name": "Stroke volume 1 Hz voltage ladder",
        "description": (f"Smallest stroke volume. {HZ} Hz, {STROKES} strokes per step, bench-supply voltage "
                        f"{VOLTAGES_MV[0]}-{VOLTAGES_MV[-1]} mV ({len(VOLTAGES_MV)} steps, operator-confirmed), "
                        f"{REST_S:.0f} s rest at 0 Hz before each drive. Microscope video on the meniscus, "
                        f"frames.csv for timing. PPK2 out of the supply path (power off)."),
        "operator": "", "pump_id": "",
    },
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": {"mode": "manual", "current_limit_ma": CURRENT_LIMIT_MA, "max_mv": 2500,
               "initial_mv": VOLTAGES_MV[0], "lead_min": 2},
    "power": {"enabled": False},
    "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
}

with open("config_stroke_volume.json", "w") as f:
    json.dump(cfg, f, indent=2)

nominal = len(VOLTAGES_MV) * (REST_S + drive_s) + REST_S
print("config_stroke_volume.json")
print(f"  {len(VOLTAGES_MV)} voltage steps: {', '.join(f'{v / 1000:g}' for v in VOLTAGES_MV)} V")
print(f"  each: confirm voltage -> {REST_S:.0f} s rest -> {drive_s:g} s at {HZ} Hz ({STROKES} strokes)")
print(f"  run time {nominal:.0f} s = {nominal / 60:.1f} min + your {len(VOLTAGES_MV)} confirmations")
