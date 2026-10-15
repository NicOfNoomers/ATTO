"""Water-prep test: does degassed water make delivery more constant?

Practical question, not a mechanism test. Pulsed delivery already holds within
3-4 % over 20 min (hold-and-floor run, 14-15 Oct), and a session calibration
removes the day-to-day offset. What is left is slow drift (about +-4 % per
10 min, worse late in long runs) and 17-23 % pulse-to-pulse scatter. If gas in
the water contributes to either, degassing the water is the cheapest fix.

RUN ORDER (one config per run, so the folder name records the water):
  1  S        air-saturated DI water, fresh fill
  2  S-sham   empty and refill with the SAME water, handled exactly as the change
              before run 3 -> what a refill alone does
  3  D        degassed DI water
  4  S2       air-saturated again -> separates the water from time drift
Water: same DI source, both at room temperature (within 2 degC).
  S: bottle half full, shake hard 1 min, stand 10 min.
  D: boil >= 10 min, fill a glass bottle to the brim, cap, cool to room
     temperature in a water bath; pour slowly down the reservoir wall; cover
     the reservoir; start the run within 10 min.

PROTOCOL per run (~34 min):
  zero 60 | 600 Hz 240 s (flush: replaces the line after a change; also shows
  the decline under continuous drive) | zero 30 | burst | zero 30
  | calibration: 60 pulses @ 100 Hz and @ 600 Hz (0.1 s every 2 s), 30 s zeros
  | hold 600 Hz, one pulse every 10 s (~3 uL/min), 2 x 5 min
  | hold 100 Hz, one pulse every 12 s (~1 uL/min), 2 x 5 min
  | burst | zero 60
Holds use fixed periods (not today's calibration), so all four runs deliver
the same pulse pattern; the analysis reads constancy, not accuracy.

WHAT DECIDES IT (compare run 3 with runs 1, 2, 4):
  drift   change of pulse volume across each 10 min hold, and calibration
          pulses vs hold pulses
  scatter pulse-to-pulse CV in calibration and holds
  level   volume per pulse; continuous flow decline over the 4 min flush
Adopt degassed water as standard prep if it lowers drift or scatter by more
than twice the refill-only change (run 2 vs run 1), and run 4 goes back.
"""
import json

PUMP_ID = ""                       # fill in
BURST_S = 72.0
RUNS = [("1 S", "air-saturated DI water, fresh fill"),
        ("2 S-sham", "same air-saturated water, reservoir emptied and refilled (refill only)"),
        ("3 D", "DEGASSED DI water (boiled, cooled sealed, room temperature)"),
        ("4 S2", "air-saturated DI water again")]
HOLDS = [(600, 10.0), (100, 12.0)]         # (Hz, pulse period s): ~3 and ~1 uL/min

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def pulses(hz, period, n):
    return {"type": "group", "repeat": n, "blocks": [sp(hz, 0.1), zero(round(period - 0.1, 2))]}


tests = [zero(60.0), sp(600, 240.0), zero(30.0), dict(burst), zero(30.0)]
tests += [pulses(100, 2.0, 60), zero(30.0), pulses(600, 2.0, 60), zero(30.0)]
for hz, per in HOLDS:
    n = int(round(300 / per))
    tests += [pulses(hz, per, n), zero(20.0), pulses(hz, per, n), zero(30.0)]
tests += [dict(burst), zero(60.0)]

for tag, water in RUNS:
    cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
           "start_time": None, "end_time": None,
           "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
           "tests": tests,
           "meta": {"name": f"Water prep {tag}",
                    "description": (f"Water-prep run {tag}: {water}. Flush 600 Hz 240 s, burst, "
                                    "60-pulse calibration at 100 and 600 Hz, 10 min holds at "
                                    "600 Hz (pulse every 10 s) and 100 Hz (every 12 s), burst."),
                    "operator": "", "pump_number": PUMP_ID},
           "serial": {"protocol": "freq_only"},
           "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
           "protocol": "freq_only"}
    fn = f"config_water_{tag.split()[0]}_{tag.split()[1].replace('-', '')}.json"
    json.dump(cfg, open(fn, "w"), indent=2)
    print("wrote", fn)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


total = sum(dur(x) for x in tests)
print(f"{len(tests)} blocks per run, {total:.0f} s = {total/60:.1f} min per run; "
      f"4 runs = {4*total/60:.0f} min plus water changes")
