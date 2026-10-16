"""Overnight, remote only: does the primed state hold, and what breaks it?

Follows the purge run of 16 Oct 20:50 (pump 260930_04, tilted outlet-up, Paper/
analysis_2026-10-16_purge): after fresh priming, 1600-2200 Hz steps unblocked the pump (600 Hz
3.5 -> 204 uL/min), the state held through 75 min of purge cycles, and the AFTER panel was the
best so far. Gas is still inside: high frequency pulls it into view. No refill is possible
tonight, so this run asks only what can be asked of a pump left as it is:
  1  Does the good state hold for hours under gentle delivery (no drive above 1000 Hz)?
  2  Does it survive rest (20 min idle, as a wearable would sit between doses)?
  3  At the end, after ~6 h: the same full panel as the purge run (direct comparison with its
     AFTER panel), then two 2000 Hz challenges to see whether more gas leaves.

CHECK (4.5 min, gentle: nothing above 1000 Hz)
  600 Hz 10 s | zero 10 | 30 x 100 Hz pulses (0.1 s / 2 s) | zero 10 | 20 x 600 Hz pulses
  (0.1 s / 2 s) | zero 10 | response 100 300 600 800 1000 Hz, 10 s each, zero 10 between | zero 30
CYCLE (64.7 min) x 6
  DELIVER  100 Hz pulses 0.1 s every 4 s for 40 min (600 pulses, ~5 uL/min) | zero 10
  IDLE     20 min zero
  CHECK
RUN: still, sync marker, CHECK 0, 6 x CYCLE, FULL PANEL (as in the purge run), 2 x CHALLENGE
(as in the purge run), sync marker, still.  7.0 h.

Liquid used: about 4 mL (delivery 1.3, checks 1.4, panel 0.8, challenges 0.4) - check the reservoir.
Video: ~7 MB per minute, ~3.2 GB; if the room lights go off at night the video is lost but the
flow and power data are not.
"""
import json

PUMP_ID = "260930_04"
BURST_S = 72.0
SYNC_GAPS = [2.0, 5.0, 3.0, 7.0, 4.0, 2.5]
RESPONSE = [100, 200, 300, 400, 600, 800, 1000, 1300, 1600, 1800, 2000, 2200, 2600, 3000]

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


def check():
    return ([sp(600, 10.0), zero(10.0), group(100, 0.1, 2.0, 30), zero(10.0),
             group(600, 0.1, 2.0, 20), zero(10.0)]
            + sum(([sp(f, 10.0), zero(10.0)] for f in (100, 300, 600, 800, 1000)), [])
            + [zero(30.0)])


def cycle():
    return [group(100, 0.1, 4.0, 600), zero(10.0), zero(1200.0)] + check()


def panel():                                  # identical to make_config_purge_attempt.panel()
    return ([zero(30.0),
             group(100, 0.1, 2.0, 150), zero(10.0),
             group(600, 0.1, 2.0, 30), zero(10.0),
             sp(100, 20.0), zero(10.0), sp(600, 60.0), zero(10.0)]
            + sum(([sp(f, 10.0), zero(10.0)] for f in RESPONSE), [])
            + [group(2000, 0.1, 2.0, 30), zero(10.0),
               sp(600, 10.0), zero(10.0),
               dict(burst), zero(30.0)])


def ref():
    return [sp(600, 10.0), zero(10.0), group(100, 0.1, 2.0, 10), zero(10.0)]


def challenge():                              # identical to the purge run's challenge
    return ref() + [zero(10.0), sp(2000, 30.0), zero(10.0), sp(600, 10.0), zero(10.0),
                    group(100, 0.1, 2.0, 60), zero(10.0)]


sections, tests = [], []


def add(name, blocks):
    sections.append((name, blocks))
    tests.extend(blocks)


add("still", [zero(60.0)])
add("sync marker", sync_marker() + [zero(10.0)])
add("CHECK 0", check())
for c in range(1, 7):
    add(f"cycle {c}: deliver 40 min, idle 20, check", cycle())
add("FULL PANEL (as purge run)", panel())
add("challenge A", challenge())
add("challenge B", challenge())
add("sync marker", sync_marker() + [zero(10.0)])
add("still", [zero(60.0)])

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": f"Overnight hold {PUMP_ID}",
                "description": ("Remote, no refill. Pump left as after the purge run (tilted outlet-up). "
                                "Check 0, then 6 x [100 Hz pulses every 4 s for 40 min, 20 min idle, gentle "
                                "check (nothing above 1000 Hz)], then the purge run's full panel and two "
                                "2000 Hz challenges. Does the primed state hold, and does rest break it?"),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_overnight_hold.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


t = 0.0
print("timeline (h:mm):")
for name, blocks in sections:
    d = sum(dur(b) for b in blocks)
    print(f"  {int(t // 3600)}:{int(t % 3600 // 60):02d}  {name:44s} {d/60:6.1f} min")
    t += d
print(f"config_overnight_hold.json: {len(tests)} blocks, {t/3600:.2f} h")
