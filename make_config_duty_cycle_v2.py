"""Burst duty-cycle config, v2: as v1 but with state brackets.

v1 (make_config_duty_cycle.py) is unchanged and still the generator of record
for the 2026-10-05 and 2026-10-07 runs. The condition grid, the alias check,
the whole-cycle windows and the two-pass OFF ordering are all carried over
verbatim -- v2 is comparable to v1 data.

WHAT CHANGED, AND WHY

v1 ran one priming burst at the start and none afterwards, so a run could not
see its own state move. Both v1 runs moved a long way:

    run          continuous 100 Hz reference, first -> last of 42
    2026-10-05   75.1 -> 38.5 uL/min   -49 %   rho -0.924  p 3e-18
    2026-10-07  110.5 -> 75.9 uL/min   -31 %   rho -0.648  p 4e-06

That decay was only visible because the continuous reference happens to repeat
42 times. It put every second-pass measurement a median 18 % below its
first-pass twin. The per-condition ratio (burst / its own reference, taken
10 s earlier) divides the drift out and stays repeatable to a median 5.4 %, so
the v1 data is usable -- but nothing in the run flagged the decay while it was
happening, and a run that died mid-way would have looked normal.

v2 adds a burst after each pass as well as before the first: three state
references at roughly 0, 50 and 100 % through the run. Pre/post ratio outside
[0.5, 2.0] is the same gate used by the bracketed flow protocol, which caught
a reservoir exhaustion (ratio 0.020) and a post-refill recovery (2.039) on
2026-09-17.

Cost: 2 extra bursts, 144 s, on a 37 min run.

Reference for reading the burst numbers on device 20260827-1: the pump has sat
at 160-270 uL/min since 21 Sep (5 Oct runs: 160-237; 7 Oct: 201). The 17 Sep
excursion to 621-717 has not repeated.
"""
import json
import math

PUMP_HZ = 100                  # drive frequency while ON; set to your operating point
ON_STROKES = [5, 20, 100]      # burst length in pump strokes (0.05 / 0.2 / 1.0 s at 100 Hz)
OFF_S = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
MEASURE_S = 30.0               # minimum burst window per condition (whole cycles)
BASE_S, REF_S = 8.0, 10.0      # 0 Hz baseline and continuous reference per condition
PASSES = 2
FS = 21.1                      # logger rate, measured
MIN_ALIAS = 1.0                # Hz; burst rates above Nyquist must not alias below this


def alias(f):
    return abs(f - round(f / FS) * FS)


zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, d: {"type": "constant", "frequency_hz": int(hz),
                    "duration_s": float(d), "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}

conditions = []
for n in ON_STROKES:
    t_on = n / PUMP_HZ
    for t_off in OFF_S:
        period = t_on + t_off
        f_b = 1.0 / period
        if f_b > FS / 2:
            assert alias(f_b) >= MIN_ALIAS, (
                f"ON {n} strokes + OFF {t_off} s: burst rate {f_b:.2f} Hz aliases to "
                f"{alias(f_b):.2f} Hz at the {FS} Hz logger rate")
        cycles = math.ceil(MEASURE_S / period)
        conditions.append(dict(strokes=n, t_on=t_on, t_off=t_off, cycles=cycles,
                               duty=t_on / period, window=cycles * period, f_b=f_b))


def condition_blocks(c):
    return [zero(BASE_S), sp(PUMP_HZ, REF_S),
            {"type": "group", "repeat": c["cycles"], "blocks": [
                sp(PUMP_HZ, c["t_on"]), zero(c["t_off"])]}]


tests = [zero(60.0), dict(burst), zero(20.0)]
for p in range(PASSES):
    asc = p % 2 == 0
    for n in (ON_STROKES if asc else ON_STROKES[::-1]):
        row = [c for c in conditions if c["strokes"] == n]
        for c in (row if asc else row[::-1]):
            tests += condition_blocks(c)
    # v2: state reference after every pass
    tests += [zero(20.0), dict(burst), zero(20.0)]
tests += [zero(BASE_S), sp(PUMP_HZ, REF_S), zero(60.0)]

cfg = {"port": "COM17", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Burst duty cycle vs wait time v2",
                "description": (f"{PUMP_HZ} Hz bursts of {ON_STROKES} strokes, OFF "
                                f"{OFF_S[0]}-{OFF_S[-1]} s, >= {MEASURE_S:.0f} s of whole cycles "
                                f"per condition, each preceded by a {BASE_S:.0f} s 0 Hz baseline "
                                f"and a {REF_S:.0f} s continuous reference. {PASSES} passes "
                                "(OFF ascending, then descending). v2 adds a priming burst after "
                                "each pass as well as before the first, so state drift is visible "
                                "within the run."),
                "operator": "", "pump_number": "20260827-1"},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}

json.dump(cfg, open("config_duty_cycle_v2.json", "w"), indent=2)

print(f"conditions at {PUMP_HZ} Hz (window = whole cycles >= {MEASURE_S:.0f} s):")
print("  strokes  ON s   OFF s  duty %  burst Hz  cycles  window s")
for c in conditions:
    note = f"  (aliases to {alias(c['f_b']):.1f} Hz)" if c["f_b"] > FS / 2 else ""
    print(f"  {c['strokes']:7d}  {c['t_on']:4.2f}  {c['t_off']:5.2f}  {100*c['duty']:6.1f}  "
          f"{c['f_b']:8.2f}  {c['cycles']:6d}  {c['window']:8.1f}{note}")

n_burst = sum(1 for t in tests if t.get("type") == "sweep")
per_pass = sum(BASE_S + REF_S + c["window"] for c in conditions)
total = sum(t["duration_s"] if t["type"] == "constant" else 72.0
            for t in tests if t["type"] in ("constant", "sweep"))
total += sum(c["window"] for c in conditions) * PASSES
print(f"\nconfig_duty_cycle_v2.json")
print(f"  {len(conditions)} conditions x {PASSES} passes, {len(tests)} top-level blocks")
print(f"  {n_burst} priming bursts (v1 had 1)")
print(f"  {per_pass/60:.1f} min per pass | total {total:.0f} s = {total/60:.1f} min")
print(f"\n  gate: post/pre burst ratio outside [0.5, 2.0] -> state moved too much,")
print(f"        treat that pass as suspect rather than averaging it in.")
