"""Generate the burst duty-cycle config: pump for T_on, wait T_off, repeat.

Question: how long can the pump wait between bursts before the flow drops too
much, and does burst pumping deliver more (or less) volume per second of
driving than continuous pumping? Driving time is the power proxy: the drive
is a 50 % PWM while on and 0 Hz while off.

Design follows the stop/start transients measured on 20260924_003618
(Lowfreq v2, 15-100 Hz dwells, n=36):
  * After stop, flow is at 50 % within ~0.09 s and 10 % by ~0.8 s; 3-4 % is
    left after 2 s. After start, 90 % is reached in ~0.3 s.
    -> OFF times span 0.02-2 s, ON times 0.05-1 s. Longer waits only show
       the pump fully off.
  * Coasting after a stop delivers ~0.29 s of full flow; the restart loses
    ~0.28 s. They cancel, so a linear system predicts burst efficiency ~1.0.
    Deviations come from nonlinear start-up (first strokes weaker or
    stronger), which short bursts expose -> ON is set in whole strokes and
    goes down to 5.
  * The logger samples at ~21.1 Hz, so ripple inside a burst cycle shorter
    than ~0.1 s is not resolved. The mean over whole cycles still is (the
    sensor's lag spreads volume in time but does not remove it); every burst
    rate above Nyquist is checked not to alias near DC.

Each condition is measured as: 0 Hz baseline -> continuous reference at the
same frequency -> burst group of whole cycles. Burst efficiency for analysis:
    (mean_burst - baseline) / (duty * (mean_reference - baseline))
1.0 = same volume per driving second as continuous pumping.
Two passes, OFF ascending then descending, so slow drift doesn't masquerade
as an OFF-time effect.

Requires the dashboard with group support (type "group").
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
tests += [zero(BASE_S), sp(PUMP_HZ, REF_S), zero(60.0)]

cfg = {"port": "COM17", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Burst duty cycle vs wait time",
                "description": (f"{PUMP_HZ} Hz bursts of {ON_STROKES} strokes, OFF "
                                f"{OFF_S[0]}-{OFF_S[-1]} s, >= {MEASURE_S:.0f} s of whole cycles "
                                f"per condition, each preceded by a {BASE_S:.0f} s 0 Hz baseline "
                                f"and a {REF_S:.0f} s continuous reference. {PASSES} passes "
                                "(OFF ascending, then descending)."),
                "operator": "", "pump_number": "20260827-1"},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}

json.dump(cfg, open("config_duty_cycle.json", "w"), indent=2)

print(f"conditions at {PUMP_HZ} Hz (window = whole cycles >= {MEASURE_S:.0f} s):")
print("  strokes  ON s   OFF s  duty %  burst Hz  cycles  window s")
for c in conditions:
    note = f"  (aliases to {alias(c['f_b']):.1f} Hz)" if c["f_b"] > FS / 2 else ""
    print(f"  {c['strokes']:7d}  {c['t_on']:4.2f}  {c['t_off']:5.2f}  {100*c['duty']:6.1f}  "
          f"{c['f_b']:8.2f}  {c['cycles']:6d}  {c['window']:8.1f}{note}")

per_pass = sum(BASE_S + REF_S + c["window"] for c in conditions)
total = 60 + 60 + 20 + PASSES * per_pass + BASE_S + REF_S + 60
print(f"\nconfig_duty_cycle.json")
print(f"  {len(conditions)} conditions x {PASSES} passes, {len(tests)} top-level blocks")
print(f"  {per_pass/60:.1f} min per pass | total {total:.0f} s = {total/60:.1f} min")
