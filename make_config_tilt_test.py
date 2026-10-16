"""Tilt test: does tilting the pump (outlet up) let the drive walk the gas out?

Idea (NM, 16 Oct): set the pump at an angle and repeat the purge test.
Why it can work: in the 15-16 Oct videos the drives move the gas a lot (it leaves the view
within 0.8 s at 1300-2000 Hz) but put it back at random, because flat on the bench nothing
gives it a direction. Buoyancy alone is too weak to free a pinned pocket at this size
(Bond number rho*g*L^2/sigma ~0.14 at 1 mm, ~0.5 at 2 mm), but while the drive shakes it
loose, a tilt gives it a preferred way to go: up, toward the outlet.

DESIGN: level -> tilted -> level in ONE run, so day-to-day drift (11-18 % between sessions)
cannot pass for a tilt effect. Each phase is built from the purge-trial unit:
  R   reference   600 Hz 10 s, zero 10 s, 10 pulses at 100 Hz (0.1 s every 2 s), zero 10 s
  I1  still       zero 10 s
  T   treatment   SHAM (zero 30 s) | F2000 (2000 Hz 30 s) | SW (1200-2600 Hz, 2 s steps)
  I2  still       zero 10 s
  R   reference
  P   100 Hz pulses every 2 s, 3 min
  I3  still       zero 10 s
  L1  level:   SHAM, F2000
  --  3 min pause: TILT the pump, outlet up (camera on the same mount, or re-frame and note it)
  T   tilted:  SHAM, F2000, SW, F2000   (repeated drive, gas has time to walk; watch the outlet)
  --  3 min pause: back to LEVEL
  L2  level:   SHAM, F2000

WHAT DECIDES IT
  primary  references in L2 vs L1 (both level, same geometry): if the tilted phase removed
           gas, L2 delivers more than L1 and its stills show less gas at the back end
  exit     gas seen in the outlet tubing during T, or air signatures in the flow trace
  compare  F2000 effect (R after minus R before) tilted vs level
References inside T are compared only with each other (the geometry changed).
Keep the reservoir level with the pump as before; tilt about the pump so its height changes
by millimetres only (1 mm of water = 0.1 mbar).
"""
import json

PUMP_ID = ""                       # fill in
BURST_S = 72.0
PAUSE_S = 180.0
SYNC_GAPS = [2.0, 5.0, 3.0, 7.0, 4.0, 2.5]

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def group(hz, on, period, n):
    return {"type": "group", "repeat": n, "blocks": [sp(hz, on), zero(round(period - on, 2))]}


def sync_marker():
    out = []
    for gap in SYNC_GAPS:
        out += [sp(600, 0.2), zero(gap)]
    return out


def ref():
    return [sp(600, 10.0), zero(10.0), group(100, 0.1, 2.0, 10), zero(10.0)]


TREAT = {"SHAM": [zero(30.0)],
         "F2000": [sp(2000, 30.0)],
         "SW": [sp(f, 2.0) for f in range(1200, 2601, 100)]}


def trial(arm):
    return ref() + [zero(10.0)] + TREAT[arm] + [zero(10.0)] + ref() + [group(100, 0.1, 2.0, 90), zero(10.0)]


sections, tests = [], []


def add(name, blocks):
    sections.append((name, blocks))
    tests.extend(blocks)


add("still", [zero(60.0)])
add("sync marker", sync_marker() + [zero(10.0)])
add("burst", [dict(burst), zero(30.0)])
for arm in ("SHAM", "F2000"):
    add(f"L1 level   {arm}", trial(arm))
add(">>> PAUSE: TILT outlet up", [zero(PAUSE_S)])
for arm in ("SHAM", "F2000", "SW", "F2000"):
    add(f"T  tilted  {arm}", trial(arm))
add(">>> PAUSE: back to LEVEL", [zero(PAUSE_S)])
for arm in ("SHAM", "F2000"):
    add(f"L2 level   {arm}", trial(arm))
add("burst", [dict(burst), zero(30.0)])
add("sync marker", sync_marker() + [zero(10.0)])
add("still", [zero(60.0)])

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Tilt test",
                "description": ("Level -> tilted (outlet up) -> level in one run. Each phase: purge-trial "
                                "units (600 Hz / 100 Hz-pulse reference before and after a 30 s treatment, "
                                "stills, 3 min 100 Hz pulse hold). Level: SHAM, F2000; tilted: SHAM, F2000, "
                                "SW, F2000; level: SHAM, F2000. 3 min pauses to tilt and to return."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_tilt_test.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


t = 0.0
print("timeline (mm:ss from start):")
for name, blocks in sections:
    d = sum(dur(b) for b in blocks)
    print(f"  {int(t // 60):3d}:{int(t % 60):02d}  {name:28s} {d/60:5.1f} min")
    t += d
print(f"config_tilt_test.json: {len(tests)} blocks, {t/60:.1f} min")
