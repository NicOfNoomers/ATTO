"""Surge check: does a pump draw the ~90 mA first phase at every drive start?

Background (Paper/analysis_2026-10-15_power, PPK2 runs 15-16 Oct, 2.5 V): on the
see-through October pump every drive start draws ~90-95 mA, then drops to ~38-46 mA after
T = 1.40 s / (1 - f / 2945 Hz), and the flow drops with it. It behaves like a self-heating
series resistance; at >= ~2600 Hz the drop never comes. NM (16 Oct) expects the original
(not see-through) RAB pump not to do this. This config tests it in ~14 min, and the same
config is written twice so the run folder names say which pump was on the rig.

Run it on the RAB pump, then swap to the see-through pump and run the other file, same
day, same supply, same tubing. Before each run, write down the pump's coil resistance
(multimeter across the pump's two leads, pump unplugged).

SECTIONS
  F1  continuous 12 s blocks at 1, 100, 600, 1000, 1600, 2000, 3000 Hz, each after 20 s off
      (first phase on the see-through pump: 1.4, 1.5, 1.8, 2.1, 3.1, 4.4 s, never at 3 kHz)
  R   recovery at 600 Hz: 5 s on, rest r, 5 s on, for r = 0.5, 1, 2, 4, 8, 16 s
  P   20 pulses at 600 Hz: 0.1 s on every 2 s (pulses never leave the first phase there)
  F2  F1 repeated (repeatability)

RUN WITH: "Measure power" ticked, VM 2.5 V, bench supply current limit 200 mA.
Analyse: python Paper/analysis_2026-10-15_power/pw9_surge_check.py <run RAB> <run see-through>
usage: python make_config_surge_check.py
"""
import json

FREQS = [1, 100, 600, 1000, 1600, 2000, 3000]
RESTS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
PUMPS = {"RAB": ("Surge check RAB pump", "original RAB pump (not see-through)"),
         "seethrough": ("Surge check see-through pump", "see-through October pump")}

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}

tests = [zero(60.0)]
for f in FREQS:                                                    # F1
    tests += [zero(20.0), sp(f, 12.0)]
for r in RESTS:                                                    # R
    tests += [zero(20.0), sp(600, 5.0), zero(r), sp(600, 5.0)]
tests += [zero(20.0), {"type": "group", "repeat": 20,              # P
                       "blocks": [sp(600, 0.1), zero(1.9)]}]
for f in FREQS:                                                    # F2
    tests += [zero(20.0), sp(f, 12.0)]
tests += [zero(30.0)]


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


total = sum(dur(x) for x in tests)
for key, (name, what) in PUMPS.items():
    cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
           "start_time": None, "end_time": None,
           "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
           "supply": {"mode": "manual", "current_limit_ma": 200, "max_mv": 2500, "initial_mv": 2500},
           "tests": tests,
           "meta": {"name": name,
                    "description": (f"PUMP: {what}. MEASURE POWER ON, 2.5 V, supply limit 200 mA. "
                                    "Does every drive start draw ~90 mA for 1.4-4.4 s? Continuous 12 s "
                                    "blocks at 1-3000 Hz (twice), 600 Hz recovery pairs (rest 0.5-16 s), "
                                    "20 x 0.1 s pulses at 600 Hz. Write the coil resistance in the notes."),
                    "operator": "", "pump_number": ""},
           "serial": {"protocol": "freq_only"},
           "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
           "protocol": "freq_only"}
    out = f"config_surge_check_{key}.json"
    json.dump(cfg, open(out, "w"), indent=2)
    print(f"{out}: {len(tests)} blocks, {total/60:.1f} min ({what})")
