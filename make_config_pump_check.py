"""Pump check without syringe control: the two ends of the PQ curve (NM, 29 Oct 2026).

Why: after the 802 mbar spike in the line (lab 29 Oct 22:53) the pump delivered 0.4-3.9 uL/min
instead of 170-250 while drawing ~30 mA. Before more PQ runs: does it still pump?

Run config_pump_check.json twice. The syringe stays idle; tick "Control pressure" so the pressure is
logged and the over-limit trip stays armed (pump to 0 Hz above Max).
  1. Outlet OPEN (syringe disconnected, line to a reservoir at outlet height): free flow at 0 mbar
     per frequency (the flow at the 0 mbar end of the PQ curve).
  2. Outlet CLOSED (syringe connected, idle): the pressure climbs to the pump's shut-off pressure
     per frequency (the other end). 22:50 suggested ~220 mbar at 400 Hz; Max 280 trips first if higher.
Drives 30 s per frequency with 15 s at 0 Hz between, so the line can relax.
"""
import json

PUMP_PORT = "COM3"
SYRINGE_PORT = "COM6"
FREQS_HZ = [200, 400, 700]
DRIVE_S = 30.0
REST_S = 15.0

tests = [{"type": "constant", "frequency_hz": 0, "duration_s": REST_S, "repeat": 1, "settle_ms": 0}]
for hz in FREQS_HZ:
    tests.append({"type": "constant", "frequency_hz": hz, "duration_s": DRIVE_S, "repeat": 1, "settle_ms": 0})
    tests.append({"type": "constant", "frequency_hz": 0, "duration_s": REST_S, "repeat": 1, "settle_ms": 0})

cfg = {
    "port": PUMP_PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Pump check", "operator": "", "pump_id": "",
             "description": "Free flow (outlet open) or shut-off pressure (outlet closed, syringe idle) at "
                            f"{', '.join(map(str, FREQS_HZ))} Hz, {DRIVE_S:g} s each. Run once per outlet state "
                            "and note which in the description."},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "pressure": {"enabled": True, "backend": "arduino", "port": SYRINGE_PORT, "firmware": "auto",
                 "sensor": {"backend": "fluigent", "channel": 1}, "limit_mbar": 280.0, "min_mbar": -30.0,
                 "microsteps": 16, "pos_max": 2470, "ul_per_step": round(1000.0 / 2470, 4)},
}
with open("config_pump_check.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"config_pump_check.json: {len(tests)} blocks, {sum(t['duration_s'] for t in tests):g} s")
