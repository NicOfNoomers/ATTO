"""Minimum-energy map of the pump at 2.5 V: the pump's version of an MPPT sweep.

Goal: for any target -- a dose (uL) or a flow (uL/min) -- find the drive setting that
needs the least energy (dose) or the least average power (flow), and know how steady it
is. The controllable settings at fixed VM are the drive frequency f, the pulse length
ON and the rest OFF. This run maps all three with the power meter on.

Basis (Paper/analysis_2026-10-15_power, runs of 15-16 Oct, 2.5 V): steady power is
repeatable (< 5 %) and rises with f (93 mW at 1-5 Hz ... 105 at 600 ... ~170 at 2000 Hz);
the first ~1.45 s of every drive draws ~2x; off draws ~0. Flow -- and so energy per uL --
depends on the pump's state, which drifts: compare within the run, not across days.

SECTIONS
  B    priming burst (state reference) at start and end
  F1   frequency map, continuous: 24 frequencies 100-3000 Hz, 12 s each (first phase +
       9 s steady), 15 s rests, ascending
  P    pulse map at PULSE_FREQS (default 600, 100, 2000 Hz), per frequency 21 conditions:
         ON 0.05/0.1/0.2/0.5/1/2/5 s at rest = 4 x ON (20 % duty) and rest = ON (50 %)
         ON 0.1 s at rest = 0.5/2/9/49 x ON;  ON 1 s at rest = 0.25/0.5/2 x ON (recovery)
       each condition >= 60 s and >= 5 cycles, preceded by a 10 s continuous reference at
       the same frequency (normalises the pump's state), conditions in shuffled order
  F2   frequency map again, descending (drift over the run; hysteresis)

RUN WITH: "Measure power" ticked, VM 2.5 V, bench supply current limit 200 mA (the
config's supply section sets the trip at 95 % = 190 mA; the first phase draws ~90 mA,
2-3 kHz is unmeasured). Check the reservoir: the estimate is printed below.
usage: python make_config_min_energy_map.py [f1 f2 ...]   (pulse-map frequencies)
"""
import json
import math
import random
import sys

PUMP_ID = ""                                       # fill in
PULSE_FREQS = [int(a) for a in sys.argv[1:]] or [600, 100, 2000]
SCAN = [100, 200, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000, 1100,
        1300, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
ONS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
BURST_S = 72.0
# rough continuous flows (uL/min, 15-16 Oct) for the liquid estimate only
Q_GUESS = {100: 65, 600: 110, 2000: 160}

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def gated(hz, on, off, min_s=60.0, min_cycles=5):
    n = max(math.ceil(min_s / (on + off)), min_cycles)
    return {"type": "group", "repeat": n, "blocks": [sp(hz, on), zero(round(off, 4))]}


def conditions():
    c = [(on, 4.0) for on in ONS] + [(on, 1.0) for on in ONS]
    c += [(0.1, r) for r in (0.5, 2.0, 9.0, 49.0)] + [(1.0, r) for r in (0.25, 0.5, 2.0)]
    return c


tests = [zero(60.0), dict(burst), zero(30.0)]
for f in SCAN:                                                  # F1
    tests += [sp(f, 12.0), zero(15.0)]
rng = random.Random(16)
plan = []
for f in PULSE_FREQS:                                           # P
    cond = conditions()
    rng.shuffle(cond)
    for on, r in cond:
        g = gated(f, on, on * r)
        tests += [zero(10.0), sp(f, 10.0), zero(10.0), g]
        plan.append((f, on, r, g["repeat"]))
    tests += [zero(20.0)]
for f in SCAN[::-1]:                                            # F2
    tests += [sp(f, 12.0), zero(15.0)]
tests += [dict(burst), zero(60.0)]

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "supply": {"mode": "manual", "current_limit_ma": 200, "max_mv": 2500, "initial_mv": 2500},
       "tests": tests,
       "meta": {"name": "Minimum energy map",
                "description": ("MEASURE POWER ON, 2.5 V, supply limit 200 mA. Frequency map 100-3000 Hz "
                                "(24 frequencies, 12 s, up at the start and down at the end); pulse map at "
                                f"{', '.join(map(str, PULSE_FREQS))} Hz: ON 0.05-5 s at 20 % and 50 % duty, "
                                "rest sweeps at ON 0.1 s and 1 s, each condition after a 10 s continuous "
                                "reference, shuffled; burst at start and end."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_min_energy_map.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


total = sum(dur(x) for x in tests)
vol = 0.0
for x in tests:
    if x["type"] == "constant" and x["frequency_hz"] > 0:
        vol += Q_GUESS.get(x["frequency_hz"], 120) * x["duration_s"] / 60
    elif x["type"] == "group":
        on_b = x["blocks"][0]
        vol += 1.3 * Q_GUESS.get(on_b["frequency_hz"], 120) * on_b["duration_s"] * x["repeat"] / 60
vol += 2 * 150
sec = {}
t = 0.0
for f in PULSE_FREQS:
    sec[f] = sum(math.ceil(max(60 / (on * (1 + r)), 5)) * on * (1 + r) + 30 for on, r in conditions()) / 60
print(f"config_min_energy_map.json: {len(tests)} blocks, {total/60:.0f} min = {total/3600:.2f} h")
print(f"  frequency map: {len(SCAN)} frequencies x 2 passes ({2*len(SCAN)*27/60:.0f} min)")
print("  pulse map: " + ", ".join(f"{f} Hz {sec[f]:.0f} min" for f in PULSE_FREQS)
      + f"; {len(plan)} conditions")
print(f"  liquid, rough estimate: {vol/1000:.1f} mL")
