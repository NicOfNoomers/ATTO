"""Long run: 20 min flow holds, then the sub-1 uL/min floor with 5 min windows.

PUMP: the October pump (NOT 20260827-1). Fill PUMP_ID before running.

BASIS: the 14 Oct flow-control run (Paper/analysis_2026-10-14_flow-control) showed
that a 0.1 s pulse delivers a fixed volume regardless of the rest after it
(periods 1-30 s): 464 nL at 600 Hz, 208 nL at 100 Hz (means over both passes,
<= 10 % duty). Flow = pulse volume x pulse rate, so pulse period = 60 * V / F.
1600 Hz is dropped: it delivered 2.0-2.6x more in pass 2 than in pass 1.

SECTION A -- HOLDS (long-term stability), 20 min each:
    600 Hz @ 3 uL/min    period 9.28 s
    600 Hz @ 30 uL/min   period 0.93 s (10 % duty, still in the linear range)
    100 Hz @ 3 uL/min    period 4.16 s
  Each hold = 4 x 5 min segments with a 20 s zero after each, 60 s zero before.
  The zeros anchor the baseline so a 20 min sensor drift is not read as pump drift.

SECTION B -- FLOOR (how low can we measure with 5 min windows), two passes:
    100 Hz @ 1.0 / 0.5 / 0.25 uL/min   (periods 12.5 / 25.0 / 49.9 s)
    600 Hz @ 1.0 / 0.5 uL/min          (periods 27.8 / 55.7 s)
    BLANK: 5 min with the pump off, analysed exactly like a step -> the floor
           (4 blanks: start and middle of pass 1, middle and end of pass 2)
  Every window >= 300 s of whole cycles and >= 5 pulses, 60 s zeros either side.
  Pass 2 runs the list in reverse.

STATE / CONTEXT: priming burst at the start, after section A, between the floor
passes and at the end; 20 s continuous 100 and 600 Hz references at the start,
between sections and at the end.
"""
import json
import math

PUMP_ID = ""                       # the October pump's ID -- fill in
ON_S = 0.1
V_NL = {100: 208.0, 600: 464.0}    # nL per 0.1 s pulse, 14 Oct
HOLDS = [(600, 3.0), (600, 30.0), (100, 3.0)]
HOLD_SEG_S, HOLD_NSEG, HOLD_GAP_S = 300.0, 4, 20.0
FLOOR = [(100, 1.0), (100, 0.5), (100, 0.25), (600, 1.0), (600, 0.5)]
WIN_S, MIN_PULSES, FLOOR_ZERO_S = 300.0, 5, 60.0
FS = 20.0
BURST_S = 72.0


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
            "repeat": 1, "settle_ms": 0}


def sp(hz, s):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
            "repeat": 1, "settle_ms": 0}


burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def period_for(hz, flow):
    return round(60.0 * V_NL[hz] * 1e-3 / flow, 2)


def pulses(hz, flow, min_s, min_n=1):
    per = period_for(hz, flow)
    off = round(per - ON_S, 2)
    n = max(math.ceil(min_s / per), min_n)
    assert 1 / per < FS / 2
    return {"type": "group", "repeat": n, "blocks": [sp(hz, ON_S), zero(off)]}, per, n


def refs():
    return [zero(15.0), sp(100, 20.0), zero(15.0), sp(600, 20.0), zero(15.0)]


plan = []
tests = [zero(60.0), dict(burst), zero(20.0)] + refs()

# ---- section A: holds --------------------------------------------------------
for hz, flow in HOLDS:
    tests.append(zero(60.0))
    for s in range(HOLD_NSEG):
        g, per, n = pulses(hz, flow, HOLD_SEG_S)
        tests += [g, zero(HOLD_GAP_S if s < HOLD_NSEG - 1 else 60.0)]
    plan.append(("hold", hz, flow, per, n, HOLD_NSEG * n * per))
tests += [zero(20.0), dict(burst), zero(20.0)] + refs()

# ---- section B: floor --------------------------------------------------------
BLANK = (0, 0.0)
pass1 = [BLANK] + FLOOR[:3] + [BLANK] + FLOOR[3:]          # 4 blanks in all:
pass2 = FLOOR[::-1][:2] + [BLANK] + FLOOR[::-1][2:] + [BLANK]  # start, mid, mid, end
for p, seq in enumerate([pass1, pass2]):
    tests.append(zero(FLOOR_ZERO_S))
    for hz, flow in seq:
        if hz == 0:
            tests.append(zero(WIN_S))                      # the blank window
            plan.append(("blank", 0, 0.0, None, None, WIN_S))
        else:
            g, per, n = pulses(hz, flow, WIN_S, MIN_PULSES)
            tests.append(g)
            plan.append((f"floor p{p+1}", hz, flow, per, n, n * per))
        tests.append(zero(FLOOR_ZERO_S))
    if p == 0:
        tests += [zero(20.0), dict(burst), zero(20.0)]
tests += [zero(20.0), dict(burst), zero(20.0)] + refs() + [zero(60.0)]

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Hold and floor",
                "description": ("A: 20 min holds at 600 Hz 3 and 30 uL/min and 100 Hz 3 uL/min "
                                "(0.1 s pulses, rate set from 14 Oct pulse volumes), 4 x 5 min "
                                "segments with 20 s zeros. B: 5 min windows at 100 Hz 1/0.5/0.25 and "
                                "600 Hz 1/0.5 uL/min plus 5 min blanks, 60 s zeros either side, two "
                                "passes. Bursts and 100/600 Hz continuous references between sections."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_hold_and_floor.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


t_acc, marks = 0.0, []
for x in tests:
    if x["type"] == "sweep":
        marks.append(t_acc / 60)
    t_acc += dur(x)
vol = sum(V_NL[x["blocks"][0]["frequency_hz"]] * 1e-3 * x["repeat"]
          for x in tests if x["type"] == "group")
vol += sum(x["duration_s"] / 60 * (70 if x["frequency_hz"] == 100 else 120)
           for x in tests if x["type"] == "constant" and x["frequency_hz"] > 0)
vol += len(marks) * 150
print("plan:")
print("   kind        Hz   target uL/min   period s   pulses   window s")
for kind, hz, flow, per, n, w in plan:
    if kind == "blank":
        print(f"   {kind:10s}   -        -             -          -     {w:7.0f}")
    elif kind == "hold":
        print(f"   {kind:10s} {hz:4d}   {flow:9.2f}      {per:7.2f}   4x{n:<4d}  4x{n*per:5.0f}")
    else:
        print(f"   {kind:10s} {hz:4d}   {flow:9.2f}      {per:7.2f}   {n:6d}   {w:7.0f}")
print(f"\nconfig_hold_and_floor.json: {len(tests)} top-level blocks, {len(marks)} bursts")
print(f"  total {t_acc:.0f} s = {t_acc/60:.0f} min = {t_acc/3600:.2f} h")
print(f"  bursts at " + ", ".join(f"{m:.0f}" for m in marks) + " min")
print(f"  liquid, rough upper estimate: {vol:.0f} uL")
