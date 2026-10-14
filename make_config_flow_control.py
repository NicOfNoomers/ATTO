"""Flow control 1-100 uL/min by duty cycle at three drive frequencies.

QUESTION
How low and how steadily can the pump deliver a set flow, using a fixed 0.1 s
pulse and a variable rest, at 100, 610 and 1100 Hz? Where does it stop working?

PUMP: the October pump, NOT 20260827-1 (that was the 17 Sep pump). Its ID is not
recorded in any run folder; fill PUMP_ID below before running.

WHY THESE FREQUENCIES (this pump's response: 15 slow step sweeps from the 5 Oct
'Pump Test' runs, Paper/analysis_2026-10-14_power-model/pm2_oct_pump_response.py)
  100 Hz   ~60 uL/min, sweep-to-sweep CV ~18 %; links to the 5/7/14 Oct duty runs
  600 Hz   ~130 uL/min, CV ~25 %; the stable band's best flow (50-700 Hz is stable)
  1600 Hz  ~350 uL/min peak, CV ~47 %; most flow per watt, least predictable
  (900-1200 Hz is a dip with CV 65-95 % -- avoided)

DESIGN
* Fixed pulse ON = 0.1 s (clear of the ~60 ms command floor); duty set by the
  rest. The 14 Oct duty map showed the result depends on duty, not pulse length.
* Duty ladder in factor-3 steps: 33, 10, 3.3, 1, 0.33 %. Continuous flow at 610
  and 1100 Hz is unknown in today's state (bursts fell 650 -> 170 since 17 Sep),
  so the ladder is wide enough to cross 1-100 uL/min for anything from ~50 to
  ~500 uL/min continuous. 100 Hz skips 0.33 % (would be ~0.3 uL/min, below LOQ).
* Window per step: whole cycles, >= 60 s and >= 4 cycles. Burst rates are all < 3.4 Hz, below the 10.55 Hz Nyquist limit.
* 15 s zero before every step (analysis drops the first 5 s: settling tail).
* 20 s continuous reference at the start and end of each frequency block:
  drift within the block, and the 100 % point of the ladder.
* Two passes: pass 2 reverses frequency order and duty order, so drift and
  order effects separate from repeatability.
* Priming burst at start, between passes and at the end (state, gate [0.5, 2]).

MEASUREMENT FLOOR (14 Oct zero blocks): sd of zero means ~0.16 uL/min,
LOD = 3 sd ~ 0.5, LOQ = 10 sd ~ 1.6 uL/min.
"""
import json
import math

FREQS = [100, 600, 1600]
PUMP_ID = ""                       # the October pump's ID -- fill in
DUTIES = {100: [1 / 3, 1 / 10, 1 / 30, 1 / 100],
          600: [1 / 3, 1 / 10, 1 / 30, 1 / 100, 1 / 300],
          1600: [1 / 3, 1 / 10, 1 / 30, 1 / 100, 1 / 300]}
ON_S = 0.1
MIN_WINDOW_S, MIN_CYCLES = 60.0, 4
ZERO_S, REF_S = 15.0, 20.0
PASSES = 2
FS = 21.1
BURST_S = 72.0                     # 60 x 1 s sweeps at the ~60 ms step floor

# continuous flow, mean -+ 1 sd of the 5 Oct sweeps; only for the printed table
Q_GUESS = {100: (49, 71), 600: (99, 165), 1600: (185, 507)}
GAIN = 1.4                         # flow per on-time at <= 33 % duty, 14 Oct map


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
            "repeat": 1, "settle_ms": 0}


def sp(hz, s):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
            "repeat": 1, "settle_ms": 0}


burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def step(hz, duty):
    off = round(ON_S * (1 / duty - 1), 3)
    period = ON_S + off
    assert 1 / period < FS / 2, "burst rate above Nyquist"
    cycles = max(math.ceil(MIN_WINDOW_S / period), MIN_CYCLES)
    grp = {"type": "group", "repeat": cycles, "blocks": [sp(hz, ON_S), zero(off)]}
    return grp, dict(hz=hz, duty=duty, off=off, period=period, cycles=cycles,
                     window=cycles * period)


tests, plan = [zero(60.0), dict(burst), zero(20.0)], []
for p in range(PASSES):
    fwd = p % 2 == 0
    for hz in (FREQS if fwd else FREQS[::-1]):
        tests += [zero(ZERO_S), sp(hz, REF_S)]
        for d in (DUTIES[hz] if fwd else DUTIES[hz][::-1]):
            g, info = step(hz, d)
            tests += [zero(ZERO_S), g]
            plan.append(dict(info, pass_=p + 1))
        tests += [zero(ZERO_S), sp(hz, REF_S)]
    tests += [zero(20.0), dict(burst), zero(20.0)]
tests += [zero(60.0)]

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Flow control 1-100 uL per min",
                "description": ("Duty-cycle flow control at 100, 600 and 1600 Hz. Fixed 0.1 s "
                                "pulses, rest set for 33/10/3.3/1/0.33 % duty, >= 60 s of whole "
                                "cycles per step, 15 s zero before each step, 20 s continuous "
                                "reference at the start and end of each frequency. Two passes, "
                                "the second reversed. Priming burst at start, mid and end."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_flow_control.json", "w"), indent=2)


def dur(t):
    if t["type"] == "constant":
        return t["duration_s"]
    if t["type"] == "sweep":
        return BURST_S
    return t["repeat"] * sum(dur(b) for b in t["blocks"])


total = sum(dur(t) for t in tests)
print("step plan, pass 1 (pass 2 = same steps, reversed):")
print("   Hz   duty %   OFF s   period s   cycles   window s   expected uL/min")
for s in [s for s in plan if s["pass_"] == 1]:
    lo, hi = (q * GAIN * s["duty"] for q in Q_GUESS[s["hz"]])
    exp = f"{lo:.1f}" if lo == hi else f"{lo:.1f}-{hi:.1f}"
    print(f"  {s['hz']:4d}   {100*s['duty']:6.2f}   {s['off']:5.2f}   {s['period']:8.2f}   "
          f"{s['cycles']:6d}   {s['window']:8.1f}   {exp}")
vol = 0.0
for t in tests:
    if t["type"] == "constant" and t["frequency_hz"] > 0:
        vol += Q_GUESS[t["frequency_hz"]][1] * t["duration_s"] / 60
    elif t["type"] == "group":
        b = t["blocks"][0]
        vol += Q_GUESS[b["frequency_hz"]][1] * GAIN * t["repeat"] * b["duration_s"] / 60
print(f"\nconfig_flow_control.json: {len(tests)} top-level blocks, "
      f"{sum(1 for t in tests if t['type'] == 'sweep')} bursts, {PASSES} passes")
print(f"  total {total:.0f} s = {total/60:.1f} min")
print(f"  liquid used, upper estimate (excl. bursts): {vol:.0f} uL")
