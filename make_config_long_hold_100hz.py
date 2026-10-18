"""Long-term delivery only: 100 Hz pulses, 0.1 s every 4 s (~5 uL/min), 24 h.

How long does delivery stay stable before it needs recalibration? Remote, same pump and water
as the overnight hold (pump 260930_04, tilted outlet-up, air-saturated lab DI water, no refill).
The overnight hold (Paper/analysis_2026-10-17_overnight-hold) showed delivery holding for 6.5 h,
600 Hz flow drifting -2.6 %/h, and the gas at the outlet end growing in steps at each check (its
600-1000 Hz blocks), not during 100 Hz pulses or rest. So this run drives ONLY the delivery
pattern: no checks, no other frequency, no bursts. The sync marker uses 100 Hz too.

RUN
  still 60 s | sync marker (100 Hz, 0.2 s pulses, irregular gaps) | 24 x [1 h of 100 Hz pulses
  0.1 s every 4 s (900 pulses) + 10 s zero (flow baseline; rest did not harm the state)] | sync
  marker | still 60 s.   24.0 h, 21,600 pulses.
WHAT IT MEASURES
  volume of every pulse (baseline = flow just before it), rolling 10 min median, drift per hour,
  time until the rolling median leaves +-5 % of the first hour's median (= time to recalibration),
  whether the drift levels off; pulse-to-pulse scatter over time; energy per pulse (PPK2); gas at
  the outlet end of the face from the video.
Liquid: ~300-390 nL per pulse -> 6.5-8.5 mL in 24 h. Check the reservoir. Video at ~7 MB/min is
~10 GB for 24 h. Stopping early keeps everything recorded so far.
"""
import json

PUMP_ID = "260930_04"
HOURS = 24
PERIOD = 4.0
SYNC_GAPS = [2.0, 5.0, 3.0, 7.0, 4.0, 2.5]

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}


def group(hz, on, period, n):
    return {"type": "group", "repeat": n, "blocks": [sp(hz, on), zero(round(period - on, 2))]}


def sync_marker():
    out = []
    for gap in SYNC_GAPS:
        out += [sp(100, 0.2), zero(gap)]
    return out


tests = [zero(60.0)] + sync_marker() + [zero(10.0)]
for h in range(HOURS):
    tests += [group(100, 0.1, PERIOD, int(3600 / PERIOD)), zero(10.0)]
tests += sync_marker() + [zero(10.0), zero(60.0)]

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": f"Long hold 100Hz 24h {PUMP_ID}",
                "description": ("Delivery only: 100 Hz pulses 0.1 s every 4 s (~5 uL/min) for 24 h in hourly "
                                "blocks with a 10 s zero between, no other frequency. Same pump and water as "
                                "the overnight hold, tilted outlet-up. Time to recalibration (+-5 %)."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_long_hold_100hz.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


t = sum(dur(x) for x in tests)
n = sum(x["repeat"] for x in tests if x["type"] == "group")
print(f"config_long_hold_100hz.json: {len(tests)} blocks, {n} pulses, {t / 3600:.2f} h")
