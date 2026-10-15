"""Session calibration, ~9 min: today's volume per 0.1 s pulse at 100 and 600 Hz.

Run it first in a session, then:
    python calibrate_pulses.py pump_runs/<this run> --targets 600:3,100:1 --hold-min 20
which prints the pulse volumes and writes config_calibrated_holds.json with
pulse rates that hit the targets on today's pump.

Design:
  zero 60 | burst (state) | zero 30
  | 60 pulses @ 100 Hz, one every 2 s | zero 30
  | 60 pulses @ 600 Hz, one every 2 s | zero 30
  | 100 Hz 20 s | zero 15 | 600 Hz 20 s | zero 30      (continuous refs, context)
60 pulses at a pulse CV of ~18-22 % give the mean to ~3 % (standard error);
the 2 s period sits inside the 1-30 s range where the pulse volume was found
independent of the rest (14 Oct). 30 s zeros bracket each group so the
baseline is anchored on both sides.
"""
import json

PUMP_ID = ""                       # fill in
FREQS = [100, 600]
ON_S, PERIOD_S, N_PULSES = 0.1, 2.0, 60
BURST_S = 72.0

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}

tests = [zero(60.0), dict(burst), zero(30.0)]
for hz in FREQS:
    tests += [{"type": "group", "repeat": N_PULSES,
               "blocks": [sp(hz, ON_S), zero(round(PERIOD_S - ON_S, 2))]}, zero(30.0)]
tests += [sp(100, 20.0), zero(15.0), sp(600, 20.0), zero(30.0)]

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Session calibration",
                "description": (f"{N_PULSES} pulses of {ON_S} s every {PERIOD_S} s at "
                                f"{' and '.join(str(f) for f in FREQS)} Hz, bracketed by zeros; "
                                "burst and continuous references for context. Analyse with "
                                "calibrate_pulses.py."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_session_calibration.json", "w"), indent=2)
total = sum(x["duration_s"] if x["type"] == "constant" else
            BURST_S if x["type"] == "sweep" else x["repeat"] * PERIOD_S for x in tests)
print(f"config_session_calibration.json: {len(tests)} blocks, {total:.0f} s = {total/60:.1f} min")
