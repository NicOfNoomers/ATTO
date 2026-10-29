"""PQ curves: pump flow vs outlet pressure, the pressure held by the stepper syringe (NM, 29 Oct 2026).

Rig: 1 mL liquid-filled syringe on the pump OUTLET, the pressure sensor (a Fluigent channel) between the
outlet and the syringe, stepper on an Arduino Uno (old sketch or firmware/syringe_v2). The flow sensor
stays where it is. For each frequency the run holds each pressure in turn; once the 1 s mean has been
within 2 mbar for 3 s it averages flow and pressure over MEASURE_S -> one row in pq_points.csv, one point
on the "PQ curve" tab (pq_curve.png at the end).

The pump flow goes into the syringe, so the syringe fills up. Before a point that would not fit, and
whenever fewer than 150 steps are left, the run sets the pump to 0 Hz, lets the pressure down and waits
for you: empty the syringe BY HAND to its empty end, reconnect, close the line, press "Done - continue".
The position is then 0 and the point is repeated. A pressure the pump cannot reach (the syringe would have
to push from its empty end and the pressure has stopped rising) is logged as "above_shutoff" (the pump's
shut-off point) and the higher pressures of that frequency are skipped.

Writes two configs:
  config_pq_smoke.json  one frequency, 3 pressures, 10 s each: checks the chain (~1.5 min nominal + clears)
  config_pq.json        the curves (~16 min nominal + settling beyond 20 s + clears)
Before the first run: Pressure panel -> port of the syringe Uno, "Find" for the pressure channel, then
Connect Manual and run Calibrate... (zero, sweep) once; the calibration is kept on the PC.
"""
import json

PUMP_PORT = "COM3"
SYRINGE_PORT = "COM6"          # the syringe Uno on the lab PC (29 Oct)
PRESSURE_CHANNEL = 1           # Fluigent index of the pressure sensor ("Find" in the Pressure panel)
MICROSTEPS = 16                # A4988 with MS1-3 high (29 Oct); at full steps one step was 10-25 mbar
FREQS_HZ = [200, 400, 700]
PRESSURES_MBAR = [0, 25, 50, 75, 100, 150, 200, 250]     # the sensor reads to 345 mbar; the limit is 280
MEASURE_S = 20.0
SETTLE_TIMEOUT_S = 120.0

PRESSURE = {
    "enabled": True, "backend": "arduino", "port": SYRINGE_PORT, "baud": 115200, "firmware": "auto",
    "sensor": {"backend": "fluigent", "channel": PRESSURE_CHANNEL},
    "limit_mbar": 280.0, "min_mbar": -100.0, "rate_mbar_s": 40.0,
    "tolerance_mbar": 2.0, "settle_s": 3.0, "microsteps": MICROSTEPS,
    # in FULL steps (multiplied by microsteps in the software)
    "pos_min": 0, "pos_max": 2470, "pos_margin": 30, "clear_reserve_steps": 150,
    # 1 mL over the old tool's 0..2470 range, per full step: only for the syringe-flow cross-check column
    "ul_per_step": round(1000.0 / 2470, 4),
}


def pq(freqs, pressures, measure_s):
    return {"type": "pq", "frequency_hz": freqs if len(freqs) > 1 else freqs[0], "pressures_mbar": pressures,
            "settle_timeout_s": SETTLE_TIMEOUT_S, "measure_s": measure_s, "on_timeout": "measure",
            "release_after": True}


def config(name, desc, tests):
    return {
        "port": PUMP_PORT, "baud": 115200, "read_timeout_s": 0.2,
        "start_time": None, "end_time": None,
        "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
        "require_pump_id": True,
        "tests": tests,
        "meta": {"name": name, "description": desc, "operator": "", "pump_id": "",
                 "syringe_ml": 1.0, "syringe_on": "outlet", "pressure_channel": PRESSURE_CHANNEL},
        "serial": {"protocol": "freq_only"},
        "protocol": "freq_only",
        "pressure": PRESSURE,
    }


smoke = config("PQ smoke test", "One PQ curve at 400 Hz, 0/25/50 mbar, 10 s each: checks syringe, pressure sensor, "
                                "flow and the clearing prompt before the full run.",
               [pq([400], [0, 25, 50], 10.0)])
full = config("PQ curves", f"Flow vs outlet pressure at {', '.join(map(str, FREQS_HZ))} Hz, "
                           f"{len(PRESSURES_MBAR)} pressures {PRESSURES_MBAR[0]}..{PRESSURES_MBAR[-1]} mbar, "
                           f"{MEASURE_S:g} s per point after settling. Syringe cleared by hand when full.",
              [pq(FREQS_HZ, PRESSURES_MBAR, MEASURE_S)])

for fname, cfg in (("config_pq_smoke.json", smoke), ("config_pq.json", full)):
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    n = sum(len(t["pressures_mbar"]) * (len(t["frequency_hz"]) if isinstance(t["frequency_hz"], list) else 1)
            for t in cfg["tests"])
    print(f"{fname}: {n} PQ points")
